#include "videogammetry/reconstructor.h"

#include <algorithm>
#include <chrono>
#include <climits>
#include <cmath>

namespace vg {

namespace {

constexpr double kRadToDeg = 57.29577951308232;

std::unique_ptr<ThreadPool> makePool(int threads) {
    const int n = threads > 0 ? threads : ThreadPool::hardwareThreads();
    return n > 1 ? std::make_unique<ThreadPool>(n) : nullptr;
}

// Angle at X between the rays to the two camera centers.
double triangulationAngleDeg(const Vec3& X, const Vec3& c1, const Vec3& c2) {
    const double c = dot(normalized(X - c1), normalized(X - c2));
    return std::acos(std::clamp(c, -1.0, 1.0)) * kRadToDeg;
}

double medianOf(std::vector<double> v) {
    if (v.empty()) return 0.0;
    auto mid = v.begin() + static_cast<std::ptrdiff_t>(v.size() / 2);
    std::nth_element(v.begin(), mid, v.end());
    return *mid;
}

}  // namespace

Reconstructor::Reconstructor(const Options& options)
    : options_(options),
      pool_(makePool(options.threads)),
      extractor_(options.processSize, options.orb, pool_.get()) {}

Reconstructor::~Reconstructor() {
    cancelMappingJob();
    cancelBackgroundBA();
}

void Reconstructor::addFrame(const FrameView& frame) {
    if (frame.data == nullptr || frame.width <= 0 || frame.height <= 0) return;
    FrameFeatures features;
    {
        auto t = profiler_.scope("extract");
        extractor_.extract(frame, static_cast<long>(frameCount_), features);
    }
    addFeatures(std::move(features));
}

void Reconstructor::addFeatures(FrameFeatures&& features) {
    processScale_ = features.processScale;
    if (frameCount_ == 0) {
        imageWidth_ = features.width;
        imageHeight_ = features.height;
        intrinsics_ = CameraIntrinsics::fromFov(features.width, features.height, options_.fovDeg);
    }
    rawKeypoints_ = std::move(features.keypoints);
    rawPositions_.resize(rawKeypoints_.size());
    keypoints_ = rawKeypoints_;
    for (std::size_t i = 0; i < keypoints_.size(); ++i) {
        rawPositions_[i] = {rawKeypoints_[i].x, rawKeypoints_[i].y};
        const Vec2 u = intrinsics_.undistortPixel(rawKeypoints_[i].x, rawKeypoints_[i].y);
        keypoints_[i].x = static_cast<float>(u.x);
        keypoints_[i].y = static_cast<float>(u.y);
    }
    descriptors_ = std::move(features.descriptors);
    colors_ = std::move(features.colors);

    previousTracked_.clear();
    for (int id : trackedPoints_) {
        if (id >= 0) previousTracked_.push_back(id);
    }
    trackedPoints_.assign(keypoints_.size(), -1);
    trackedCount_ = 0;
    lastWasKeyframe_ = false;

    if (state_ == State::Initializing) {
        auto t = profiler_.scope("init");
        tryInitialize();
    } else {
        bool ok;
        {
            auto t = profiler_.scope("track");
            ok = track();
        }
        if (ok) {
            state_ = State::Tracking;
            lostCount_ = 0;
            ++stats_.trackedFrames;
            velocity_ = previousPose_.inverse().then(pose_);
            hasVelocity_ = true;
            previousPose_ = pose_;

            const Keyframe& last = keyframes_.back();
            const int lastTracked = static_cast<int>(
                std::count_if(last.pointIds.begin(), last.pointIds.end(), [](int id) { return id >= 0; }));
            const long gap = static_cast<long>(frameCount_) - lastKeyframeFrame_;
            const MappingParams& mp = options_.mapping;
            const bool viewChanged = trackedCount_ < mp.keyframeTrackedRatio * lastTracked ||
                                     trackedCount_ < 2 * mp.minTrackedPoints;
            if (gap >= mp.minKeyframeGap && (viewChanged || gap >= mp.maxKeyframeGap)) {
                auto t = profiler_.scope("mapping");
                addKeyframe();
            }
        } else {
            state_ = State::Lost;
            ++stats_.lostFrames;
            hasVelocity_ = false;
            if (++lostCount_ > options_.mapping.maxLostFrames) startNewMap();
        }
    }
    if (options_.mapping.mappingThread) pollMappingJob(false);
    pollBackgroundBA();
    ++frameCount_;
}

void Reconstructor::tryInitialize() {
    const MappingParams& mp = options_.mapping;
    FrameData current;
    current.index = static_cast<long>(frameCount_);
    current.keypoints = keypoints_;
    current.rawPositions = rawPositions_;
    current.descriptors = descriptors_;
    current.colors = colors_;
    selectInitFeatures(current);

    if (reference_.index < 0 || static_cast<int>(reference_.keypoints.size()) < mp.minInitialPoints) {
        reference_ = std::move(current);
        return;
    }

    // Match against the reference over the whole image: the motion between
    // the two frames is unknown and may be large. Only the selected features
    // take part; matches are mapped back to the full feature lists.
    MatchParams matchParams = options_.match;
    matchParams.searchRadius = 0;
    std::vector<Match> matches;
    matchFeatures(current.initKeypoints, current.initDescriptors, reference_.initKeypoints,
                  reference_.initDescriptors, matchParams, matches, nullptr, pool_.get());
    for (Match& m : matches) {
        m.query = current.initIndex[m.query];
        m.train = reference_.initIndex[m.train];
    }

    const long age = current.index - reference_.index;
    if (static_cast<int>(matches.size()) < mp.minInitialPoints || age > mp.maxReferenceAge) {
        // The view has moved on (or too much time passed): start over from here.
        if (matches.size() < 0.3 * reference_.initKeypoints.size() || age > mp.maxReferenceAge) {
            reference_ = std::move(current);
        }
        return;
    }

    std::vector<Vec2> p1(matches.size()), p2(matches.size());
    for (std::size_t i = 0; i < matches.size(); ++i) {
        const Keypoint& a = reference_.keypoints[matches[i].train];
        const Keypoint& b = current.keypoints[matches[i].query];
        p1[i] = intrinsics_.normalize(a.x, a.y);
        p2[i] = intrinsics_.normalize(b.x, b.y);
    }
    TwoViewParams tv = options_.twoView;
    tv.seed += static_cast<std::uint32_t>(frameCount_);
    const TwoViewResult r = estimateRelativePose(p1, p2, intrinsics_.fx, tv, pool_.get());
    if (!r.ok || r.medianParallaxDeg < mp.minInitParallaxDeg) return;  // keep waiting for parallax

    // Triangulate the inliers; keep well-conditioned points only.
    const double fx = intrinsics_.fx;
    const Pose second{r.R, r.t};
    const Vec3 c2 = second.center();
    std::vector<Vec3> X(matches.size());
    std::vector<char> good(matches.size(), 0);
    std::vector<double> depths;
    for (std::size_t i = 0; i < matches.size(); ++i) {
        if (!r.inlier[i]) continue;
        if (!triangulate(p1[i], p2[i], r.R, r.t, X[i])) continue;
        if (X[i].z <= 0 || second.transform(X[i]).z <= 0) continue;
        if (triangulationAngleDeg(X[i], {}, c2) < mp.minTriangulationAngleDeg) continue;
        if (reprojectionErrorPx(Pose{}, X[i], p1[i], fx) > mp.maxReprojectionErrorPx) continue;
        if (reprojectionErrorPx(second, X[i], p2[i], fx) > mp.maxReprojectionErrorPx) continue;
        good[i] = 1;
        depths.push_back(X[i].z);
    }
    if (static_cast<int>(depths.size()) < mp.minInitialPoints) return;

    // Fix the arbitrary monocular scale: median scene depth = 1.
    const double s = 1.0 / medianOf(depths);

    keyframes_.clear();
    points_.clear();
    Keyframe k1, k2;
    k1.frameIndex = reference_.index;
    k1.keypoints = reference_.keypoints;
    k1.rawPositions = reference_.rawPositions;
    k1.descriptors = reference_.descriptors;
    k1.colors = reference_.colors;
    k1.pointIds.assign(k1.keypoints.size(), -1);
    k2.frameIndex = current.index;
    k2.pose = {second.R, second.t * s};
    k2.keypoints = current.keypoints;
    k2.rawPositions = current.rawPositions;
    k2.descriptors = current.descriptors;
    k2.colors = current.colors;
    k2.pointIds.assign(k2.keypoints.size(), -1);

    recentPoints_.clear();
    keyframes_.push_back(std::move(k1));
    keyframes_.push_back(std::move(k2));
    for (std::size_t i = 0; i < matches.size(); ++i) {
        if (!good[i]) continue;
        MapPoint p;
        p.position = X[i] * s;
        p.descriptor = keyframes_[1].descriptors[matches[i].query];
        p.level = keyframes_[1].keypoints[matches[i].query].level;
        p.color = keyframes_[0].colors[matches[i].train];
        p.firstKeyframe = 1;
        createPoint(std::move(p), {0, matches[i].train}, {1, matches[i].query});
    }

    pose_ = keyframes_[1].pose;
    previousPose_ = pose_;
    hasVelocity_ = false;  // the pair may span many frames; no per-frame velocity yet
    trackedPoints_ = keyframes_[1].pointIds;
    trackedCount_ = static_cast<int>(depths.size());
    lastKeyframeFrame_ = current.index;
    lastWasKeyframe_ = true;
    reference_ = FrameData{};
    state_ = State::Tracking;
    lostCount_ = 0;
    ++stats_.maps;
}

void Reconstructor::selectInitFeatures(FrameData& frame) const {
    const int n = static_cast<int>(frame.keypoints.size());
    const int cap = options_.mapping.initFeatures;
    frame.initIndex.clear();
    if (cap <= 0 || n <= cap) {
        // All of them, in their own order (the result is then exactly as
        // without a cap).
        frame.initIndex.resize(static_cast<std::size_t>(n));
        for (int i = 0; i < n; ++i) frame.initIndex[i] = i;
    } else {
        // Strongest first in each cell of a grid of about cap / 8 cells,
        // taken round-robin over the cells, so the selection covers the image.
        const int side = std::max(1, static_cast<int>(std::lround(std::sqrt(cap / 8.0))));
        const float w = static_cast<float>(std::max(1, imageWidth_)), h = static_cast<float>(std::max(1, imageHeight_));
        std::vector<std::vector<int>> cells(static_cast<std::size_t>(side) * side);
        for (int i = 0; i < n; ++i) {
            const Keypoint& k = frame.keypoints[i];
            const int cx = std::clamp(static_cast<int>(k.x / w * side), 0, side - 1);
            const int cy = std::clamp(static_cast<int>(k.y / h * side), 0, side - 1);
            cells[static_cast<std::size_t>(cy) * side + cx].push_back(i);
        }
        for (std::vector<int>& c : cells) {
            std::stable_sort(c.begin(), c.end(), [&](int a, int b) {
                return frame.keypoints[a].response > frame.keypoints[b].response;
            });
        }
        for (std::size_t round = 0; static_cast<int>(frame.initIndex.size()) < cap; ++round) {
            bool any = false;
            for (const std::vector<int>& c : cells) {
                if (round >= c.size()) continue;
                any = true;
                frame.initIndex.push_back(c[round]);
                if (static_cast<int>(frame.initIndex.size()) >= cap) break;
            }
            if (!any) break;
        }
        std::sort(frame.initIndex.begin(), frame.initIndex.end());  // keep the frame's own order
    }
    frame.initKeypoints.resize(frame.initIndex.size());
    frame.initDescriptors.resize(frame.initIndex.size());
    for (std::size_t j = 0; j < frame.initIndex.size(); ++j) {
        frame.initKeypoints[j] = frame.keypoints[frame.initIndex[j]];
        frame.initDescriptors[j] = frame.descriptors[frame.initIndex[j]];
    }
}

std::vector<std::pair<int, int>> Reconstructor::covisibleKeyframes(int k) const {
    std::vector<int> shared(keyframes_.size(), 0);
    std::vector<int> touched;
    for (int id : keyframes_[k].pointIds) {
        if (id < 0) continue;
        for (const PointObservation& o : points_[id].observations) {
            if (o.keyframe == k) continue;
            if (shared[o.keyframe]++ == 0) touched.push_back(o.keyframe);
        }
    }
    std::vector<std::pair<int, int>> result;
    result.reserve(touched.size());
    for (int other : touched) result.push_back({other, shared[other]});
    // Most shared first; ties: the more recent keyframe.
    std::sort(result.begin(), result.end(), [](const auto& a, const auto& b) {
        return a.second != b.second ? a.second > b.second : a.first > b.first;
    });
    return result;
}

std::vector<int> Reconstructor::localKeyframes() const {
    const MappingParams& mp = options_.mapping;
    const int nk = static_cast<int>(keyframes_.size());
    std::vector<int> result;
    if (nk == 0) return result;
    std::vector<char> chosen(static_cast<std::size_t>(nk), 0);
    const auto add = [&](int k) {
        if (!chosen[k] && static_cast<int>(result.size()) < mp.localKeyframes) {
            chosen[k] = 1;
            result.push_back(k);
        }
    };

    // 1. Keyframes that see the points tracked in the previous frame, by
    //    number of those points they see.
    std::vector<int> votes(static_cast<std::size_t>(nk), 0);
    for (int id : previousTracked_) {
        if (id < 0 || id >= static_cast<int>(points_.size()) || points_[id].bad) continue;
        for (const PointObservation& o : points_[id].observations) ++votes[o.keyframe];
    }
    std::vector<std::pair<int, int>> seeing;  // (votes, keyframe)
    for (int k = 0; k < nk; ++k) {
        if (votes[k] > 0) seeing.push_back({votes[k], k});
    }
    std::sort(seeing.begin(), seeing.end(), [](const auto& a, const auto& b) {
        return a.first != b.first ? a.first > b.first : a.second > b.second;
    });
    add(nk - 1);  // the newest keyframe always
    for (const auto& [v, k] : seeing) add(k);

    // 2. Their best covisible neighbours (strongest first); after a tracking
    //    loss, the newest keyframe's.
    std::vector<int> seeds;
    for (const auto& [v, k] : seeing) seeds.push_back(k);
    if (seeds.empty()) seeds.push_back(nk - 1);
    for (int k : seeds) {
        if (static_cast<int>(result.size()) >= mp.localKeyframes) break;
        const std::vector<std::pair<int, int>> covisible = covisibleKeyframes(k);
        for (int i = 0; i < static_cast<int>(covisible.size()) && i < mp.covisibleNeighbours; ++i) {
            add(covisible[i].first);
        }
    }
    return result;
}

std::vector<int> Reconstructor::localPoints() const {
    std::vector<char> seen(points_.size(), 0);
    std::vector<int> ids;
    for (int k : localKeyframes()) {
        for (int id : keyframes_[k].pointIds) {
            if (id >= 0 && !seen[id] && !points_[id].bad) {
                seen[id] = 1;
                ids.push_back(id);
            }
        }
    }
    return ids;
}

int Reconstructor::searchByProjection(const Pose& pose, float radius, const std::vector<int>& local,
                                      std::vector<int>& pointOfKeypoint) const {
    constexpr int kMaxDistance = 64;
    constexpr float kRatio = 0.9f;
    constexpr int kMaxLevelDiff = 3;

    std::vector<char> alreadyMatched(points_.size(), 0);
    for (int id : pointOfKeypoint) {
        if (id >= 0) alreadyMatched[id] = 1;
    }
    const KeypointGrid grid(keypoints_, std::max(radius, 8.0f));
    const float r2 = radius * radius;

    struct Candidate {
        int keypoint = -1;
        int distance = INT_MAX;
    };
    std::vector<Candidate> best(local.size());

    parallelFor(pool_.get(), 0, static_cast<int>(local.size()), 64, [&](int j0, int j1) {
        for (int j = j0; j < j1; ++j) {
            const int id = local[j];
            if (alreadyMatched[id]) continue;
            const MapPoint& p = points_[id];
            const Vec3 c = pose.transform(p.position);
            if (c.z <= 1e-6) continue;
            const float u = static_cast<float>(intrinsics_.fx * c.x / c.z + intrinsics_.cx);
            const float v = static_cast<float>(intrinsics_.fy * c.y / c.z + intrinsics_.cy);
            if (u < 0 || v < 0 || u >= imageWidth_ || v >= imageHeight_) continue;

            int bestKp = -1, bestDist = INT_MAX, second = INT_MAX;
            grid.forEachNear(u, v, radius, [&](int k) {
                if (pointOfKeypoint[k] >= 0) return;
                const Keypoint& kp = keypoints_[k];
                const float dx = kp.x - u, dy = kp.y - v;
                if (dx * dx + dy * dy > r2 || std::abs(kp.level - p.level) > kMaxLevelDiff) return;
                const int d = hammingDistance(p.descriptor, descriptors_[k]);
                if (d < bestDist || (d == bestDist && k < bestKp)) {
                    second = bestDist;
                    bestDist = d;
                    bestKp = k;
                } else if (d < second) {
                    second = d;
                }
            });
            if (bestKp >= 0 && bestDist <= kMaxDistance && (second == INT_MAX || bestDist < kRatio * second)) {
                best[j] = {bestKp, bestDist};
            }
        }
    });

    // One map point per keypoint: keep the closest descriptor (ties: first in order).
    std::vector<int> keypointDistance(keypoints_.size(), INT_MAX);
    std::vector<int> assigned(keypoints_.size(), -1);
    for (std::size_t j = 0; j < local.size(); ++j) {
        const Candidate& c = best[j];
        if (c.keypoint >= 0 && c.distance < keypointDistance[c.keypoint]) {
            keypointDistance[c.keypoint] = c.distance;
            assigned[c.keypoint] = local[j];
        }
    }
    int count = 0;
    for (std::size_t k = 0; k < assigned.size(); ++k) {
        if (assigned[k] >= 0) {
            pointOfKeypoint[k] = assigned[k];
            ++count;
        }
    }
    return count;
}

int Reconstructor::solveAndRefine(const Pose& prior, const std::vector<int>& local,
                                  std::vector<int>& pointOfKeypoint, Pose& pose) {
    const double fx = intrinsics_.fx;
    std::vector<Vec3> world;
    std::vector<Vec2> obs;
    std::vector<int> kpIndex;
    const auto collect = [&] {
        world.clear();
        obs.clear();
        kpIndex.clear();
        for (std::size_t k = 0; k < pointOfKeypoint.size(); ++k) {
            if (pointOfKeypoint[k] < 0) continue;
            world.push_back(points_[pointOfKeypoint[k]].position);
            obs.push_back(intrinsics_.normalize(keypoints_[k].x, keypoints_[k].y));
            kpIndex.push_back(static_cast<int>(k));
        }
    };

    collect();
    if (world.size() < 4) return 0;
    PnPParams params = options_.pnp;
    params.seed += static_cast<std::uint32_t>(frameCount_);
    const PnPResult r = solvePnPRansac(world, obs, fx, params, &prior, pool_.get());
    if (!r.ok) return 0;
    for (std::size_t i = 0; i < kpIndex.size(); ++i) {
        if (!r.inlier[i]) pointOfKeypoint[kpIndex[i]] = -1;
    }
    pose = r.pose;

    // With an accurate pose, look for more map points in a small radius and
    // refine on everything found.
    searchByProjection(pose, 6.0f, local, pointOfKeypoint);
    collect();
    refinePose(world, obs, nullptr, fx, params.inlierThresholdPx, 10, pose);

    int inliers = 0;
    for (std::size_t i = 0; i < kpIndex.size(); ++i) {
        if (reprojectionErrorPx(pose, world[i], obs[i], fx) <= params.inlierThresholdPx) {
            ++inliers;
        } else {
            pointOfKeypoint[kpIndex[i]] = -1;
        }
    }
    return inliers;
}

bool Reconstructor::track() {
    if (keyframes_.empty()) return false;
    const MappingParams& mp = options_.mapping;
    const Keyframe& lastKf = keyframes_.back();
    const Pose prior = hasVelocity_ ? previousPose_.then(velocity_) : previousPose_;
    const std::vector<int> local = localPoints();

    std::vector<int> pok(keypoints_.size(), -1);
    Pose pose;
    int tracked = 0;

    // 1. Projection search around the predicted pose; wider if motion was fast.
    int found = searchByProjection(prior, mp.projectionRadiusPx, local, pok);
    if (found < mp.minTrackedPoints) {
        ++stats_.wideSearches;
        std::fill(pok.begin(), pok.end(), -1);
        found = searchByProjection(prior, mp.wideRadiusPx, local, pok);
    }
    if (found >= mp.minTrackedPoints) tracked = solveAndRefine(prior, local, pok, pose);

    // 2. Prediction useless (sudden motion, or recovering from lost): match the
    //    whole image against the last keyframe and inherit its map points.
    if (tracked < mp.minTrackedPoints) {
        ++stats_.keyframeSearches;
        std::fill(pok.begin(), pok.end(), -1);
        MatchParams matchParams = options_.match;
        matchParams.searchRadius = 0;
        std::vector<Match> matches;
        matchFeatures(keypoints_, descriptors_, lastKf.keypoints, lastKf.descriptors, matchParams, matches,
                      nullptr, pool_.get());
        for (const Match& m : matches) {
            const int id = lastKf.pointIds[m.train];
            if (id >= 0 && !points_[id].bad) pok[m.query] = id;
        }
        tracked = solveAndRefine(lastKf.pose, local, pok, pose);
    }
    if (tracked < mp.minTrackedPoints) return false;

    pose_ = pose;
    trackedPoints_ = std::move(pok);
    trackedCount_ = tracked;
    return true;
}

void Reconstructor::addKeyframe() {
    // A new keyframe builds on a refined map, as when local BA runs inline:
    // merge the local adjustment in flight first (usually finished by now; it
    // has had the frames since the last keyframe). Its successor starts below,
    // for this keyframe, once triangulation and fusion are done. A focal
    // refinement (a much larger window, every few keyframes, intrinsics only
    // change slowly) is not waited for: it is merged whenever it finishes.
    if (options_.mapping.mappingThread) {
        auto t = profiler_.scope("kf.ba wait");
        pollMappingJob(!mappingJobFocal_, false);
        localBAPending_ = false;  // superseded by this keyframe's
    }
    Keyframe kf;
    kf.frameIndex = static_cast<long>(frameCount_);
    kf.pose = pose_;
    kf.keypoints = keypoints_;
    kf.rawPositions = rawPositions_;
    kf.descriptors = descriptors_;
    kf.colors = colors_;
    kf.pointIds.assign(kf.keypoints.size(), -1);
    const int newIndex = static_cast<int>(keyframes_.size());
    keyframes_.push_back(std::move(kf));
    for (std::size_t k = 0; k < trackedPoints_.size(); ++k) {
        const int id = trackedPoints_[k];
        if (id < 0 || points_[id].bad) continue;
        addObservation(id, {newIndex, static_cast<int>(k)});
        MapPoint& p = points_[id];
        p.descriptor = keyframes_[newIndex].descriptors[k];  // keep the newest view's appearance
        p.level = keyframes_[newIndex].keypoints[k].level;
    }
    lastKeyframeFrame_ = static_cast<long>(frameCount_);
    lastWasKeyframe_ = true;

    {
        auto t = profiler_.scope("kf.cull");
        cullRecentPoints();
    }

    // Triangulation partners: of the keyframes sharing the most points with
    // the new one, those whose baseline relative to the scene depth gives
    // enough parallax (most shared first).
    const MappingParams& mp = options_.mapping;
    const int n = static_cast<int>(keyframes_.size());
    const Keyframe& newest = keyframes_[n - 1];
    const double depth = medianSceneDepth(newest);
    const std::vector<std::pair<int, int>> covisible = covisibleKeyframes(n - 1);
    std::vector<int> partners;
    for (int i = 0; i < static_cast<int>(covisible.size()) && i < mp.triangulationSearch; ++i) {
        const int k = covisible[i].first;
        const double baseline = norm(keyframes_[k].pose.center() - newest.pose.center());
        const double ratio = depth > 0 ? baseline / depth : 0.0;
        if (ratio >= mp.minBaselineRatio) partners.push_back(k);
        if (static_cast<int>(partners.size()) >= mp.triangulationKeyframes) break;
    }
    {
        auto t = profiler_.scope("kf.triangulate");
        for (int k : partners) triangulateBetween(k, n - 1);
    }
    {
        auto t = profiler_.scope("kf.fuse");
        searchInNeighbors(n - 1);
    }

    const bool focalDue = mp.refineFocal && !backgroundOwnsIntrinsics_ && mp.focalEveryKeyframes > 0 &&
                          n % mp.focalEveryKeyframes == 0 && n >= mp.focalBAKeyframes;
    if (mp.mappingThread) {
        // Solved on the mapping thread; this frame goes on without waiting.
        if (mp.localBAKeyframes > 0) localBAPending_ = true;
        if (focalDue) focalBAPending_ = true;
        pollMappingJob(false);
    } else if (mp.localBAKeyframes > 0) {
        // The new keyframe and the keyframes sharing the most points with it
        // (counted after triangulation, which adds shared points).
        std::vector<int> local{n - 1};
        for (const auto& [k, shared] : covisibleKeyframes(n - 1)) {
            if (static_cast<int>(local.size()) >= mp.localBAKeyframes) break;
            local.push_back(k);
        }
        auto t = profiler_.scope("kf.local ba");
        const BAReport r = adjust(local, mp.localBAIterations);
        if (r.ok) {
            ++stats_.localBAs;
            stats_.localBARmsBefore += r.initialRmsPx;
            stats_.localBARmsAfter += r.finalRmsPx;
            stats_.localBAOutliers += r.outliers;
            // The current frame is the newest keyframe: continue from its refined pose.
            pose_ = keyframes_.back().pose;
            previousPose_ = pose_;
        }
    }
    if (!mp.mappingThread && focalDue) {
        std::vector<int> window{n - 1};
        for (const auto& [k, shared] : covisibleKeyframes(n - 1)) {
            if (static_cast<int>(window.size()) >= mp.focalBAKeyframes) break;
            window.push_back(k);
        }
        auto t = profiler_.scope("kf.focal ba");
        const BAReport r = adjust(window, mp.localBAIterations, true);
        if (r.ok) {
            pose_ = keyframes_.back().pose;
            previousPose_ = pose_;
        }
    }
    trackedPoints_ = keyframes_.back().pointIds;
}

double Reconstructor::medianSceneDepth(const Keyframe& kf) const {
    std::vector<double> depths;
    for (int id : kf.pointIds) {
        if (id < 0) continue;
        const double z = kf.pose.transform(points_[id].position).z;
        if (z > 0) depths.push_back(z);
    }
    return medianOf(std::move(depths));
}

BAReport Reconstructor::adjust(const std::vector<int>& keyframes, int iterations, bool refineFocal) {
    std::unique_ptr<BAJob> job = buildJob(keyframes, iterations, refineFocal);
    if (!job) return {};
    job->report = bundleAdjust(job->problem, job->focal, job->params, pool_.get());
    return applyJob(*job);
}

std::unique_ptr<Reconstructor::BAJob> Reconstructor::buildJob(const std::vector<int>& keyframes, int iterations,
                                                              bool refineFocal) const {
    const int nk = static_cast<int>(keyframes_.size());
    if (nk < 2 || keyframes.empty()) return nullptr;
    std::vector<char> variable(static_cast<std::size_t>(nk), 0);
    for (int k : keyframes) variable[k] = 1;

    auto job = std::make_unique<BAJob>();
    job->mapEpoch = mapEpoch_;
    job->focal = intrinsics_.fx;
    job->keyframeCount = nk;
    job->pointCount = static_cast<int>(points_.size());

    // Points seen by the keyframes being adjusted.
    std::vector<int> pointIndex(points_.size(), -1);
    for (int k : keyframes) {
        for (int id : keyframes_[k].pointIds) {
            if (id >= 0 && !points_[id].bad && pointIndex[id] < 0) {
                pointIndex[id] = static_cast<int>(job->pointIds.size());
                job->pointIds.push_back(id);
            }
        }
    }
    if (job->pointIds.empty()) return nullptr;

    // Cameras: every keyframe that sees those points (the adjusted ones plus
    // fixed neighbours), in keyframe order so the reduced system stays banded.
    // With the focal free, only the adjusted keyframes (others were built with
    // the old focal and would hold it back).
    std::vector<char> involved(static_cast<std::size_t>(nk), 0);
    if (refineFocal) {
        involved = variable;
    } else {
        for (int id : job->pointIds) {
            for (const PointObservation& o : points_[id].observations) involved[o.keyframe] = 1;
        }
    }
    BAProblem& problem = job->problem;
    for (int k = 0; k < nk; ++k) {
        if (!involved[k]) continue;
        const Keyframe& kf = keyframes_[k];
        const int camera = static_cast<int>(job->cameraKeyframe.size());
        for (std::size_t i = 0; i < kf.pointIds.size(); ++i) {
            const int id = kf.pointIds[i];
            if (id < 0 || points_[id].bad || pointIndex[id] < 0) continue;
            problem.observations.push_back({camera, pointIndex[id],
                                            intrinsics_.normalize(kf.rawPositions[i].x, kf.rawPositions[i].y),
                                            kf.keypoints[i].scale});
            job->obsRef.push_back({k, static_cast<int>(i)});
        }
        job->cameraKeyframe.push_back(k);
        problem.poses.push_back(kf.pose);
        problem.fixedPose.push_back(variable[k] ? 0 : 1);
    }
    // Gauge: monocular maps are defined up to position, rotation and scale.
    // Without fixed neighbours, fix the two oldest keyframes.
    if (std::count(problem.fixedPose.begin(), problem.fixedPose.end(), 1) == 0) {
        for (std::size_t c = 0; c < problem.fixedPose.size() && c < 2; ++c) problem.fixedPose[c] = 1;
    }
    for (int id : job->pointIds) problem.points.push_back(points_[id].position);

    BAParams& params = job->params;
    params.maxIterations = iterations;
    params.outlierThresholdPx = options_.mapping.maxReprojectionErrorPx + 0.5;
    params.k1 = intrinsics_.k1;
    params.refineFocal = refineFocal;
    params.focalPriorSigma = options_.mapping.focalPriorSigma;
    params.refineDistortion = refineFocal && options_.mapping.refineDistortion;
    params.distortionPriorSigma = options_.mapping.distortionPriorSigma;
    return job;
}

BAReport Reconstructor::applyJob(BAJob& job) {
    BAReport& report = job.report;
    if (!report.ok || job.mapEpoch != mapEpoch_) {
        report.ok = false;
        return report;
    }
    // Solved with intrinsics that have since been refined: outdated.
    if (!job.params.refineFocal && (std::abs(intrinsics_.fx / job.focal - 1) > 0.01 ||
                                    std::abs(intrinsics_.k1 - job.params.k1) > 0.01)) {
        ++stats_.backgroundDiscarded;
        report.ok = false;
        return report;
    }
    if (job.params.refineFocal) {
        const bool determined = report.focalStd <= options_.mapping.focalMaxStd &&
                                (!job.params.refineDistortion || report.k1Std <= options_.mapping.distortionMaxStd);
        if (!determined) {
            ++stats_.focalRejected;  // not determined by this motion: change nothing
            report.ok = false;
            return report;
        }
        ++stats_.focalUpdates;
        intrinsics_.fx = job.focal * report.focalScale;
        intrinsics_.fy = intrinsics_.fx;
        intrinsics_.k1 = report.k1;
    }

    // New keyframe poses: the solution for the job's variable cameras, the
    // current pose for its fixed ones.
    const int nk = static_cast<int>(keyframes_.size());
    std::vector<Pose> before(static_cast<std::size_t>(nk));
    for (int k = 0; k < nk; ++k) before[k] = keyframes_[k].pose;
    std::vector<char> placed(static_cast<std::size_t>(nk), 0);
    for (std::size_t c = 0; c < job.cameraKeyframe.size(); ++c) {
        const int k = job.cameraKeyframe[c];
        if (!job.problem.fixedPose[c]) keyframes_[k].pose = job.problem.poses[c];
        placed[k] = 1;
    }
    // Keyframes created while the job ran keep their pose relative to their
    // most covisible placed keyframe (older keyframes outside the job stay).
    for (int k = 0; k < std::min(nk, job.keyframeCount); ++k) placed[k] = 1;
    const auto carry = [&](const Pose& pose, int parent) {
        // pose' = pose o parent_before^-1 o parent_after
        return keyframes_[parent].pose.then(before[parent].inverse()).then(pose);
    };
    for (int k = 0; k < nk; ++k) {
        if (placed[k]) continue;
        for (const auto& [other, shared] : covisibleKeyframes(k)) {
            if (placed[other]) {
                keyframes_[k].pose = carry(before[k], other);
                break;
            }
        }
        placed[k] = 1;
    }

    // Points: the solution for the job's points; others move with the first
    // keyframe that sees them.
    std::vector<char> solved(points_.size(), 0);
    for (std::size_t j = 0; j < job.pointIds.size(); ++j) {
        MapPoint& p = points_[job.pointIds[j]];
        if (p.bad) continue;  // removed or merged meanwhile
        p.position = job.problem.points[j];
        solved[job.pointIds[j]] = 1;
    }
    for (std::size_t id = static_cast<std::size_t>(job.pointCount); id < points_.size(); ++id) {
        MapPoint& p = points_[id];
        if (p.bad || solved[id] || p.observations.empty()) continue;
        const int k = p.observations.front().keyframe;
        p.position = keyframes_[k].pose.inverse().transform(before[k].transform(p.position));
    }

    // Rejected observations, if still linked to the same point.
    for (std::size_t o = 0; o < job.obsRef.size(); ++o) {
        if (report.inlier[o]) continue;
        const PointObservation& ref = job.obsRef[o];
        const int id = job.pointIds[job.problem.observations[o].point];
        if (keyframes_[ref.keyframe].pointIds[ref.keypoint] == id) removeObservation(id, ref);
    }

    // The current frame keeps its pose relative to the newest keyframe.
    const int last = nk - 1;
    const Pose correction = keyframes_[last].pose.then(before[last].inverse());
    pose_ = correction.then(pose_);
    previousPose_ = correction.then(previousPose_);

    if (job.params.refineFocal) undistortAll();
    return report;
}

void Reconstructor::cancelMappingJob() {
    localBAPending_ = focalBAPending_ = false;
    if (!mappingJob_) return;
    cancelMapping_ = true;
    mappingDone_.wait();
    mappingJob_.reset();
    cancelMapping_ = false;
}

void Reconstructor::pollMappingJob(bool wait, bool startPending) {
    using Clock = std::chrono::steady_clock;
    if (mappingJob_) {
        if (!wait && mappingDone_.wait_for(std::chrono::seconds(0)) != std::future_status::ready) return;
        mappingDone_.get();
        const auto t0 = Clock::now();
        const BAReport r = applyJob(*mappingJob_);
        if (!mappingJobFocal_ && r.ok) {
            ++stats_.localBAs;
            stats_.localBARmsBefore += r.initialRmsPx;
            stats_.localBARmsAfter += r.finalRmsPx;
            stats_.localBAOutliers += r.outliers;
        }
        // Solve time on the mapping thread (*), merge time on this one.
        profiler_.add(mappingJobFocal_ ? "kf.focal ba*" : "kf.local ba*", mappingJob_->solveMs);
        mappingJob_.reset();
        profiler_.add("kf.ba merge", std::chrono::duration<double, std::milli>(Clock::now() - t0).count());
    }
    if (!startPending || state_ != State::Tracking || keyframes_.size() < 2) return;
    if (focalBAPending_) {
        focalBAPending_ = false;
        startMappingJob(true);
    } else if (localBAPending_) {
        localBAPending_ = false;
        startMappingJob(false);
    }
}

void Reconstructor::startMappingJob(bool focal) {
    using Clock = std::chrono::steady_clock;
    const MappingParams& mp = options_.mapping;
    const int n = static_cast<int>(keyframes_.size());
    // The newest keyframe and the keyframes sharing the most points with it.
    const int size = focal ? mp.focalBAKeyframes : mp.localBAKeyframes;
    std::vector<int> window{n - 1};
    for (const auto& [k, shared] : covisibleKeyframes(n - 1)) {
        if (static_cast<int>(window.size()) >= size) break;
        window.push_back(k);
    }
    {
        auto t = profiler_.scope("kf.ba build");
        mappingJob_ = buildJob(window, mp.localBAIterations, focal);
    }
    if (!mappingJob_) return;
    mappingJobFocal_ = focal;
    mappingJob_->params.cancel = &cancelMapping_;
    if (!mappingPool_) {
        const int threads = mp.mappingThreads > 0 ? mp.mappingThreads : std::max(1, ThreadPool::hardwareThreads() / 2);
        if (threads > 1) mappingPool_ = std::make_unique<ThreadPool>(threads);
    }
    BAJob* job = mappingJob_.get();
    ThreadPool* pool = mappingPool_.get();
    mappingDone_ = std::async(std::launch::async, [job, pool] {
        const auto start = Clock::now();
        job->report = bundleAdjust(job->problem, job->focal, job->params, pool);
        job->solveMs = std::chrono::duration<double, std::milli>(Clock::now() - start).count();
    });
}

void Reconstructor::cancelBackgroundBA() {
    if (!backgroundJob_) return;
    cancelBackground_ = true;
    backgroundDone_.wait();
    backgroundJob_.reset();
    cancelBackground_ = false;
}

void Reconstructor::pollBackgroundBA() {
    using Clock = std::chrono::steady_clock;
    const MappingParams& mp = options_.mapping;
    if (backgroundJob_) {
        if (backgroundDone_.wait_for(std::chrono::seconds(0)) != std::future_status::ready) return;
        backgroundDone_.get();
        const auto t0 = Clock::now();
        const bool withIntrinsics = backgroundJob_->params.refineFocal;
        const CameraIntrinsics local = intrinsics_;
        if (applyJob(*backgroundJob_).ok) {
            ++stats_.backgroundPasses;
            stats_.backgroundLastMs = backgroundJob_->solveMs;
            // Once the whole-map estimate agrees with the local one (both have
            // converged), the whole-map passes own the intrinsics.
            if (withIntrinsics && std::abs(intrinsics_.fx / local.fx - 1) < 0.005 &&
                std::abs(intrinsics_.k1 - local.k1) < 0.005) {
                backgroundOwnsIntrinsics_ = true;
            }
        } else if (withIntrinsics && backgroundJob_->report.iterations > 0) {
            backgroundIntrinsicsFailed_ = true;  // not determined: next pass keeps them fixed
        }
        backgroundJob_.reset();
        stats_.backgroundSnapshotMs += std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
    }
    if (!mp.backgroundBA || state_ != State::Tracking ||
        static_cast<int>(keyframes_.size()) < mp.backgroundBAMinKeyframes) {
        return;
    }

    // Next pass over the whole map, refining the intrinsics with it (jointly
    // with the whole map, which also repairs parts built with old intrinsics),
    // unless the last attempt showed the motion cannot determine them.
    const auto t0 = Clock::now();
    std::vector<int> all(keyframes_.size());
    for (std::size_t k = 0; k < all.size(); ++k) all[k] = static_cast<int>(k);
    const bool intrinsics = mp.refineFocal && !backgroundIntrinsicsFailed_;
    backgroundIntrinsicsFailed_ = false;
    backgroundJob_ = buildJob(all, mp.backgroundBAIterations, intrinsics);
    stats_.backgroundSnapshotMs += std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
    if (!backgroundJob_) return;
    backgroundJob_->params.cancel = &cancelBackground_;
    if (!backgroundPool_) {
        // Low priority: it only uses CPU time that tracking, decoding and
        // feature extraction leave over.
        const int n = mp.backgroundBAThreads > 0 ? mp.backgroundBAThreads
                                                 : std::max(1, ThreadPool::hardwareThreads() / 2);
        if (n > 1) backgroundPool_ = std::make_unique<ThreadPool>(n, true);
    }
    BAJob* job = backgroundJob_.get();
    ThreadPool* pool = backgroundPool_.get();
    backgroundDone_ = std::async(std::launch::async, [job, pool] {
        lowerCurrentThreadPriority();
        const auto start = Clock::now();
        job->report = bundleAdjust(job->problem, job->focal, job->params, pool);
        job->solveMs = std::chrono::duration<double, std::milli>(Clock::now() - start).count();
    });
}

void Reconstructor::undistortAll() {
    for (Keyframe& kf : keyframes_) {
        for (std::size_t i = 0; i < kf.keypoints.size(); ++i) {
            const Vec2 u = intrinsics_.undistortPixel(kf.rawPositions[i].x, kf.rawPositions[i].y);
            kf.keypoints[i].x = static_cast<float>(u.x);
            kf.keypoints[i].y = static_cast<float>(u.y);
        }
    }
    for (std::size_t i = 0; i < keypoints_.size(); ++i) {
        const Vec2 u = intrinsics_.undistortPixel(rawPositions_[i].x, rawPositions_[i].y);
        keypoints_[i].x = static_cast<float>(u.x);
        keypoints_[i].y = static_cast<float>(u.y);
    }
}

int Reconstructor::createPoint(MapPoint point, PointObservation a, PointObservation b) {
    const int id = static_cast<int>(points_.size());
    point.observations.clear();
    points_.push_back(std::move(point));
    addObservation(id, a);
    addObservation(id, b);
    recentPoints_.push_back(id);
    ++stats_.pointsCreated;
    return id;
}

void Reconstructor::addObservation(int pointId, PointObservation obs) {
    keyframes_[obs.keyframe].pointIds[obs.keypoint] = pointId;
    points_[pointId].observations.push_back(obs);
}

void Reconstructor::removeObservation(int pointId, PointObservation obs) {
    MapPoint& p = points_[pointId];
    int& slot = keyframes_[obs.keyframe].pointIds[obs.keypoint];
    if (slot == pointId) slot = -1;
    const auto it = std::find_if(p.observations.begin(), p.observations.end(), [&](const PointObservation& o) {
        return o.keyframe == obs.keyframe && o.keypoint == obs.keypoint;
    });
    if (it != p.observations.end()) p.observations.erase(it);
    if (p.observations.size() < 2) setBad(pointId);
}

void Reconstructor::setBad(int pointId) {
    MapPoint& p = points_[pointId];
    for (const PointObservation& o : p.observations) {
        int& slot = keyframes_[o.keyframe].pointIds[o.keypoint];
        if (slot == pointId) slot = -1;
    }
    p.observations.clear();
    p.bad = true;
}

void Reconstructor::mergePoints(int keep, int drop) {
    if (keep == drop || points_[keep].bad || points_[drop].bad) return;
    MapPoint& k = points_[keep];

    // Verify: drop's observations must agree with keep's position (the
    // projection test that found the duplicate covered one keyframe only).
    // Mostly disagreeing: two different points, or the map has drifted
    // between them -- leave both.
    const auto fits = [&](const PointObservation& o) {
        const Keyframe& kf = keyframes_[o.keyframe];
        const Keypoint& kp = kf.keypoints[o.keypoint];
        const double err = reprojectionErrorPx(kf.pose, k.position, intrinsics_.normalize(kp.x, kp.y), intrinsics_.fx);
        return err * err <= 5.991 * kp.scale * kp.scale;
    };
    const std::vector<PointObservation> dropObs = points_[drop].observations;
    const auto agreeing = std::count_if(dropObs.begin(), dropObs.end(), fits);
    if (2 * agreeing < static_cast<long>(dropObs.size())) return;

    for (const PointObservation& o : dropObs) {
        int& slot = keyframes_[o.keyframe].pointIds[o.keypoint];
        const bool keepSeen = std::any_of(k.observations.begin(), k.observations.end(),
                                          [&](const PointObservation& ko) { return ko.keyframe == o.keyframe; });
        if (keepSeen || !fits(o)) {
            if (slot == drop) slot = -1;  // the keypoint is freed
        } else {
            slot = keep;
            k.observations.push_back(o);
        }
    }
    points_[drop].observations.clear();
    points_[drop].bad = true;
    ++stats_.pointsMerged;

    // Appearance from the most recent keyframe that sees it: tracking matches
    // against current frames, and an old view's descriptor may no longer fit.
    const PointObservation* newest = nullptr;
    for (const PointObservation& o : k.observations) {
        if (!newest || o.keyframe > newest->keyframe) newest = &o;
    }
    if (newest) {
        k.descriptor = keyframes_[newest->keyframe].descriptors[newest->keypoint];
        k.level = keyframes_[newest->keyframe].keypoints[newest->keypoint].level;
    }
}

std::vector<Reconstructor::FuseCandidate> Reconstructor::fuseCandidates(const std::vector<int>& pointIds) const {
    std::vector<Vec3> centers(keyframes_.size());
    std::vector<char> centerKnown(keyframes_.size(), 0);
    for (int id : pointIds) {
        for (const PointObservation& o : points_[id].observations) {
            if (!centerKnown[o.keyframe]) {
                centers[o.keyframe] = keyframes_[o.keyframe].pose.center();
                centerKnown[o.keyframe] = 1;
            }
        }
    }
    std::vector<FuseCandidate> out(pointIds.size());
    parallelFor(pool_.get(), 0, static_cast<int>(pointIds.size()), 128, [&](int j0, int j1) {
        for (int j = j0; j < j1; ++j) {
            const MapPoint& p = points_[pointIds[j]];
            FuseCandidate& c = out[j];
            c.id = pointIds[j];
            c.observers.reserve(p.observations.size());
            c.logDistLevel.reserve(p.observations.size());
            Vec3 sum;
            for (const PointObservation& o : p.observations) {
                const Vec3 d0 = p.position - centers[o.keyframe];
                const double n = norm(d0);
                if (n <= 0) continue;
                sum = sum + d0 * (1.0 / n);
                c.observers.push_back(o.keyframe);
                c.logDistLevel.push_back({std::log(n), keyframes_[o.keyframe].keypoints[o.keypoint].level});
            }
            c.meanDir = normalized(sum);
        }
    });
    return out;
}

void Reconstructor::fuseIntoKeyframe(int k, const std::vector<FuseCandidate>& candidates) {
    const MappingParams& mp = options_.mapping;
    const Keyframe& kf = keyframes_[k];
    const Vec3 center = kf.pose.center();
    const double cosMaxView = std::cos(mp.maxViewAngleDeg / kRadToDeg);
    const float scaleFactor = options_.orb.scaleFactor;
    const double logScale = std::log(static_cast<double>(scaleFactor));
    const int maxLevel = std::max(options_.orb.levels - 1, 0);
    const KeypointGrid grid(kf.keypoints, 16.0f);

    // Best keypoint for each point (read-only, in parallel), applied in order.
    std::vector<int> bestKeypoint(candidates.size(), -1);
    parallelFor(pool_.get(), 0, static_cast<int>(candidates.size()), 64, [&](int j0, int j1) {
        for (int j = j0; j < j1; ++j) {
            const FuseCandidate& cand = candidates[j];
            if (std::find(cand.observers.begin(), cand.observers.end(), k) != cand.observers.end()) continue;
            const MapPoint& p = points_[cand.id];
            if (p.bad || cand.logDistLevel.empty()) continue;
            const Vec3 ray = p.position - center;
            const double dist = norm(ray);
            if (dist <= 0 || dot(ray, cand.meanDir) < cosMaxView * dist) continue;

            // The pyramid level the point should appear at from here: a
            // feature seen at level L from distance d0 appears about
            // log(d0 / d) / log(scaleFactor) levels higher from d (predicted
            // from the observation at the most similar distance).
            const double logDist = std::log(dist);
            double bestLogRatio = 1e30;
            int predictedLevel = p.level;
            for (const auto& [logD0, level] : cand.logDistLevel) {
                const double logRatio = logD0 - logDist;
                if (std::abs(logRatio) < std::abs(bestLogRatio)) {
                    bestLogRatio = logRatio;
                    predictedLevel = static_cast<int>(std::lround(level + logRatio / logScale));
                }
            }
            predictedLevel = std::clamp(predictedLevel, 0, maxLevel);

            const Vec3 c = kf.pose.transform(p.position);
            if (c.z <= 1e-6) continue;
            const float u = static_cast<float>(intrinsics_.fx * c.x / c.z + intrinsics_.cx);
            const float v = static_cast<float>(intrinsics_.fy * c.y / c.z + intrinsics_.cy);
            if (u < 0 || v < 0 || u >= imageWidth_ || v >= imageHeight_) continue;

            const float radius = mp.fuseRadiusPx * std::pow(scaleFactor, static_cast<float>(predictedLevel));
            int best = -1, bestDist = INT_MAX, second = INT_MAX;
            grid.forEachNear(u, v, radius, [&](int i) {
                const Keypoint& kp = kf.keypoints[i];
                if (std::abs(kp.level - predictedLevel) > 1) return;
                const float dx = kp.x - u, dy = kp.y - v;
                // Reprojection within the 95% chi-square bound (2 dof) at the keypoint's scale.
                if (dx * dx + dy * dy > 5.991f * kp.scale * kp.scale) return;
                const int d = hammingDistance(p.descriptor, kf.descriptors[i]);
                if (d < bestDist) {
                    second = bestDist;
                    bestDist = d;
                    best = i;
                } else if (d < second) {
                    second = d;
                }
            });
            // Unambiguous only: repeated texture (windows, paving) has look-alikes nearby.
            if (best >= 0 && bestDist <= mp.fuseMaxHamming &&
                (second == INT_MAX || bestDist < mp.fuseRatio * second)) {
                bestKeypoint[j] = best;
            }
        }
    });

    for (std::size_t j = 0; j < candidates.size(); ++j) {
        const int kp = bestKeypoint[j];
        const int id = candidates[j].id;
        if (kp < 0 || points_[id].bad) continue;
        if (std::any_of(points_[id].observations.begin(), points_[id].observations.end(),
                        [&](const PointObservation& o) { return o.keyframe == k; })) {
            continue;  // linked meanwhile (e.g. through a merge)
        }
        const int existing = keyframes_[k].pointIds[kp];
        if (existing >= 0 && !points_[existing].bad) {
            if (!options_.mapping.fuseMerge) continue;
            // Two points for one feature: keep the better observed one.
            if (points_[existing].observations.size() >= points_[id].observations.size()) {
                mergePoints(existing, id);
            } else {
                mergePoints(id, existing);
            }
        } else {
            addObservation(id, {k, kp});
            ++stats_.observationsFused;
        }
    }
}

void Reconstructor::searchInNeighbors(int k) {
    const MappingParams& mp = options_.mapping;
    if (mp.fuseNeighbours <= 0) return;

    // Neighbours: most covisible keyframes, and their best neighbours.
    std::vector<char> chosen(keyframes_.size(), 0);
    chosen[k] = 1;
    std::vector<int> targets;
    const std::vector<std::pair<int, int>> first = covisibleKeyframes(k);
    for (int i = 0; i < static_cast<int>(first.size()) && i < mp.fuseNeighbours; ++i) {
        chosen[first[i].first] = 1;
        targets.push_back(first[i].first);
    }
    const std::size_t firstOrder = targets.size();
    for (std::size_t i = 0; i < firstOrder; ++i) {
        const std::vector<std::pair<int, int>> second = covisibleKeyframes(targets[i]);
        for (int j = 0, added = 0; j < static_cast<int>(second.size()) && added < mp.fuseSecondOrder; ++j) {
            const int t = second[j].first;
            if (chosen[t]) continue;
            chosen[t] = 1;
            targets.push_back(t);
            ++added;
        }
    }

    // Forward: the new keyframe's points into each neighbour.
    {
        std::vector<int> ids;
        for (int id : keyframes_[k].pointIds) {
            if (id >= 0 && !points_[id].bad) ids.push_back(id);
        }
        const std::vector<FuseCandidate> candidates = fuseCandidates(ids);
        for (int t : targets) fuseIntoKeyframe(t, candidates);
    }

    // Backward: the neighbours' points into the new keyframe.
    std::vector<char> collected(points_.size(), 0);
    std::vector<int> ids;
    for (int t : targets) {
        for (int id : keyframes_[t].pointIds) {
            if (id >= 0 && !points_[id].bad && !collected[id]) {
                collected[id] = 1;
                ids.push_back(id);
            }
        }
    }
    fuseIntoKeyframe(k, fuseCandidates(ids));
}

void Reconstructor::cullRecentPoints() {
    const MappingParams& mp = options_.mapping;
    const int newest = static_cast<int>(keyframes_.size()) - 1;
    std::size_t kept = 0;
    for (int id : recentPoints_) {
        MapPoint& p = points_[id];
        if (p.bad) continue;
        const int age = newest - p.firstKeyframe;
        if (age >= mp.probationKeyframes) {
            if (static_cast<int>(p.observations.size()) < mp.minConfirmingKeyframes) {
                setBad(id);
                ++stats_.pointsCulled;
            }
            continue;  // decided: off probation
        }
        recentPoints_[kept++] = id;
    }
    recentPoints_.resize(kept);
}

void Reconstructor::finish() {
    // Merge the local BA in flight; the final global BA below supersedes the
    // pending ones and the background pass.
    localBAPending_ = focalBAPending_ = false;
    pollMappingJob(true);
    cancelBackgroundBA();
    const int iterations = options_.mapping.globalBAIterations;
    if (iterations <= 0 || keyframes_.size() < 2) return;
    auto t = profiler_.scope("global ba");
    std::vector<int> all(keyframes_.size());
    for (std::size_t k = 0; k < all.size(); ++k) all[k] = static_cast<int>(k);
    stats_.globalBA = adjust(all, iterations, options_.mapping.refineFocal);
    if (!stats_.globalBA.ok && options_.mapping.refineFocal) {
        stats_.globalBA = adjust(all, iterations);  // focal not determined: refine the rest only
    }
    stats_.globalBACameras = static_cast<int>(keyframes_.size());
    stats_.globalBAPoints = static_cast<int>(std::count_if(points_.begin(), points_.end(),
                                                           [](const MapPoint& p) { return !p.bad; }));
}

int Reconstructor::triangulateBetween(int olderIndex, int newerIndex) {
    const Keyframe& older = keyframes_[olderIndex];
    const Keyframe& newer = keyframes_[newerIndex];
    const MappingParams& mp = options_.mapping;
    const double fx = intrinsics_.fx;

    // Relative pose older -> newer is known. A match x1 (older) <-> x2 (newer)
    // must satisfy x2^T E x1 = 0: x1 lies on the epipolar line E^T x2.
    const Pose rel = older.pose.inverse().then(newer.pose);
    const Mat3 Et = (skew(rel.t) * rel.R).transposed();
    const Vec3 c2 = rel.center();
    const Pose olderInv = older.pose.inverse();

    // Features not yet attached to map points.
    std::vector<int> ia, ib;
    for (std::size_t i = 0; i < older.pointIds.size(); ++i) {
        if (older.pointIds[i] < 0) ia.push_back(static_cast<int>(i));
    }
    for (std::size_t i = 0; i < newer.pointIds.size(); ++i) {
        if (newer.pointIds[i] < 0) ib.push_back(static_cast<int>(i));
    }
    std::vector<Vec2> xa(ia.size());
    for (std::size_t j = 0; j < ia.size(); ++j) {
        xa[j] = intrinsics_.normalize(older.keypoints[ia[j]].x, older.keypoints[ia[j]].y);
    }

    // Guided search: for each newer feature, the best older candidate near
    // its epipolar line, if unambiguous among those candidates.
    struct Candidate {
        int older = -1;  // index into ia
        int distance = INT_MAX;
    };
    std::vector<Candidate> best(ib.size());
    const int maxLevelDiff = options_.match.maxLevelDiff;

    // The older features go into a grid (normalized image coordinates), so
    // each epipolar line only visits the cells its band crosses instead of
    // every feature: the search no longer grows with the square of the
    // feature count. The band is as wide as the widest allowed distance
    // (coarsest level); each candidate still gets the exact test below.
    double minX = 0, maxX = 0, minY = 0, maxY = 0, maxScale = 1;
    for (std::size_t a = 0; a < ia.size(); ++a) {
        if (a == 0 || xa[a].x < minX) minX = xa[a].x;
        if (a == 0 || xa[a].x > maxX) maxX = xa[a].x;
        if (a == 0 || xa[a].y < minY) minY = xa[a].y;
        if (a == 0 || xa[a].y > maxY) maxY = xa[a].y;
        maxScale = std::max<double>(maxScale, older.keypoints[ia[a]].scale);
    }
    const int grid = std::clamp(static_cast<int>(std::sqrt(ia.size() / 8.0)), 1, 256);  // ~8 features per cell
    const double cellW = std::max(maxX - minX, 1e-9) / grid, cellH = std::max(maxY - minY, 1e-9) / grid;
    // Clamped before the cast: a band edge can lie far outside the image.
    const auto cellX = [&](double x) { return static_cast<int>(std::clamp((x - minX) / cellW, 0.0, grid - 1.0)); };
    const auto cellY = [&](double y) { return static_cast<int>(std::clamp((y - minY) / cellH, 0.0, grid - 1.0)); };
    std::vector<int> cellStart(static_cast<std::size_t>(grid) * grid + 1, 0), cellItems(ia.size());
    for (std::size_t a = 0; a < ia.size(); ++a) ++cellStart[cellY(xa[a].y) * grid + cellX(xa[a].x) + 1];
    for (std::size_t c = 1; c < cellStart.size(); ++c) cellStart[c] += cellStart[c - 1];
    {
        std::vector<int> fill(cellStart.begin(), cellStart.end() - 1);
        for (std::size_t a = 0; a < ia.size(); ++a) {
            cellItems[fill[cellY(xa[a].y) * grid + cellX(xa[a].x)]++] = static_cast<int>(a);
        }
    }
    // Widest band half-width in normalized units, with a margin for rounding.
    const double band = mp.epipolarDistancePx * maxScale / fx * (1.0 + 1e-6) + 1e-12;

    parallelFor(pool_.get(), 0, static_cast<int>(ib.size()), 32, [&](int j0, int j1) {
        for (int j = j0; j < j1; ++j) {
            const Keypoint& kb = newer.keypoints[ib[j]];
            const Vec2 x2 = intrinsics_.normalize(kb.x, kb.y);
            const Vec3 line = Et * Vec3{x2.x, x2.y, 1.0};
            const double lineNorm = std::sqrt(line.x * line.x + line.y * line.y);
            if (lineNorm < 1e-12) continue;  // no translation: no epipolar line
            const double toPx = fx / lineNorm;
            const Descriptor& db = newer.descriptors[ib[j]];
            int bestA = -1, bestDist = INT_MAX, second = INT_MAX;
            // Ties go to the lowest index, so the result does not depend on
            // the order the cells are visited in.
            const auto consider = [&](int a) {
                const Keypoint& ka = older.keypoints[ia[a]];
                if (std::abs(ka.level - kb.level) > maxLevelDiff) return;
                const double lineDistPx = std::abs(line.x * xa[a].x + line.y * xa[a].y + line.z) * toPx;
                if (lineDistPx > mp.epipolarDistancePx * ka.scale) return;
                const int d = hammingDistance(db, older.descriptors[ia[a]]);
                if (d < bestDist || (d == bestDist && a < bestA)) {
                    second = bestDist;
                    bestDist = d;
                    bestA = a;
                } else if (d < second) {
                    second = d;
                }
            };
            const auto visitCell = [&](int cx, int cy) {
                const int c = cy * grid + cx;
                for (int i = cellStart[c]; i < cellStart[c + 1]; ++i) consider(cellItems[i]);
            };
            // Walk along the line's dominant axis; in each column (row) of
            // cells the band covers a range of rows (columns).
            if (std::abs(line.y) >= std::abs(line.x)) {
                const double half = band * lineNorm / std::abs(line.y);  // band's vertical half-height
                for (int cx = 0; cx < grid; ++cx) {
                    const double x0 = minX + cx * cellW, x1 = x0 + cellW;
                    const double y0 = -(line.x * x0 + line.z) / line.y, y1 = -(line.x * x1 + line.z) / line.y;
                    const double lo = std::min(y0, y1) - half, hi = std::max(y0, y1) + half;
                    if (hi < minY || lo > maxY) continue;
                    for (int cy = cellY(lo), cyEnd = cellY(hi); cy <= cyEnd; ++cy) visitCell(cx, cy);
                }
            } else {
                const double half = band * lineNorm / std::abs(line.x);  // band's horizontal half-width
                for (int cy = 0; cy < grid; ++cy) {
                    const double y0 = minY + cy * cellH, y1 = y0 + cellH;
                    const double x0 = -(line.y * y0 + line.z) / line.x, x1 = -(line.y * y1 + line.z) / line.x;
                    const double lo = std::min(x0, x1) - half, hi = std::max(x0, x1) + half;
                    if (hi < minX || lo > maxX) continue;
                    for (int cx = cellX(lo), cxEnd = cellX(hi); cx <= cxEnd; ++cx) visitCell(cx, cy);
                }
            }
            if (bestA < 0 || bestDist > mp.triangulationMaxHamming) continue;
            if (second != INT_MAX && bestDist >= mp.triangulationRatio * second) continue;
            best[j] = {bestA, bestDist};
        }
    });

    // Mutual best: one newer feature per older feature (closest descriptor).
    std::vector<int> ownerOfOlder(ia.size(), -1);
    for (std::size_t j = 0; j < ib.size(); ++j) {
        const Candidate& c = best[j];
        if (c.older < 0) continue;
        const int owner = ownerOfOlder[c.older];
        if (owner < 0 || c.distance < best[owner].distance) ownerOfOlder[c.older] = static_cast<int>(j);
    }

    // Rotation consistency: the change of keypoint orientation between the two
    // keyframes should agree for all matches (one camera roll); keep the
    // three most common 12-degree bins, as ORB-SLAM does.
    constexpr int kBins = 30;
    constexpr double kTwoPi = 6.283185307179586;
    std::vector<int> binOf(ib.size(), -1);
    int histogram[kBins] = {};
    for (std::size_t a = 0; a < ia.size(); ++a) {
        const int j = ownerOfOlder[a];
        if (j < 0) continue;
        double d = newer.keypoints[ib[j]].angle - older.keypoints[ia[a]].angle;
        d -= kTwoPi * std::floor(d / kTwoPi);
        const int bin = std::min(kBins - 1, static_cast<int>(d / kTwoPi * kBins));
        binOf[j] = bin;
        ++histogram[bin];
    }
    int top[3] = {-1, -1, -1};
    for (int bin = 0; bin < kBins; ++bin) {
        for (int r = 0; r < 3; ++r) {
            if (top[r] < 0 || histogram[bin] > histogram[top[r]]) {
                for (int s = 2; s > r; --s) top[s] = top[s - 1];
                top[r] = bin;
                break;
            }
        }
    }
    // Like ORB-SLAM, secondary bins count only if reasonably populated.
    const auto keptBin = [&](int bin) {
        if (bin == top[0]) return true;
        for (int r = 1; r < 3; ++r) {
            if (bin == top[r] && top[0] >= 0 && histogram[bin] >= 0.1 * histogram[top[0]]) return true;
        }
        return false;
    };

    int created = 0;
    for (std::size_t aj = 0; aj < ia.size(); ++aj) {
        const int j = ownerOfOlder[aj];
        if (j < 0 || !keptBin(binOf[j])) continue;
        const int a = ia[aj], b = ib[j];
        const Keypoint& ka = older.keypoints[a];
        const Keypoint& kb = newer.keypoints[b];
        const Vec2 x1 = xa[aj];
        const Vec2 x2 = intrinsics_.normalize(kb.x, kb.y);
        Vec3 X;
        if (!triangulate(x1, x2, rel.R, rel.t, X)) continue;
        if (X.z <= 0 || rel.transform(X).z <= 0) continue;
        if (triangulationAngleDeg(X, {}, c2) < mp.minTriangulationAngleDeg) continue;
        // Coarser pyramid levels localize keypoints less precisely.
        if (reprojectionErrorPx(Pose{}, X, x1, fx) > mp.maxReprojectionErrorPx * ka.scale) continue;
        if (reprojectionErrorPx(rel, X, x2, fx) > mp.maxReprojectionErrorPx * kb.scale) continue;

        MapPoint p;
        p.position = olderInv.transform(X);
        p.descriptor = newer.descriptors[b];
        p.level = kb.level;
        p.color = newer.colors[b];
        p.firstKeyframe = newerIndex;
        createPoint(std::move(p), {olderIndex, a}, {newerIndex, b});
        ++created;
    }
    return created;
}

void Reconstructor::startNewMap() {
    if (points_.size() > bestPoints_.size()) {
        bestPoints_ = points_;
        bestTrajectory_.clear();
        for (const Keyframe& k : keyframes_) bestTrajectory_.push_back(k.pose.center());
    }
    keyframes_.clear();
    points_.clear();
    recentPoints_.clear();
    ++mapEpoch_;  // a running background pass belongs to the old map
    cancelMappingJob();  // so does the local BA in flight
    backgroundOwnsIntrinsics_ = false;
    reference_ = FrameData{};
    state_ = State::Initializing;
    hasVelocity_ = false;
    lostCount_ = 0;
}

double Reconstructor::fovDeg() const {
    if (intrinsics_.fx <= 0) return options_.fovDeg;
    return 2 * std::atan(0.5 * std::max(imageWidth_, imageHeight_) / intrinsics_.fx) * kRadToDeg;
}

PointCloud Reconstructor::pointCloud() const {
    const std::vector<MapPoint>& src = points_.size() >= bestPoints_.size() ? points_ : bestPoints_;
    PointCloud cloud;
    cloud.points.reserve(src.size());
    for (const MapPoint& p : src) {
        if (p.bad) continue;
        Point q;
        q.x = static_cast<float>(p.position.x);
        q.y = static_cast<float>(p.position.y);
        q.z = static_cast<float>(p.position.z);
        q.r = p.color[0];
        q.g = p.color[1];
        q.b = p.color[2];
        cloud.points.push_back(q);
    }
    return cloud;
}

std::vector<Vec3> Reconstructor::trajectory() const {
    if (points_.size() >= bestPoints_.size()) {
        std::vector<Vec3> t;
        for (const Keyframe& k : keyframes_) t.push_back(k.pose.center());
        return t;
    }
    return bestTrajectory_;
}

}  // namespace vg
