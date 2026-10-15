#include "videogammetry/reconstructor.h"

#include <algorithm>
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
    keypoints_ = std::move(features.keypoints);
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
    ++frameCount_;
}

void Reconstructor::tryInitialize() {
    const MappingParams& mp = options_.mapping;
    FrameData current;
    current.index = static_cast<long>(frameCount_);
    current.keypoints = keypoints_;
    current.descriptors = descriptors_;
    current.colors = colors_;

    if (reference_.index < 0 || static_cast<int>(reference_.keypoints.size()) < mp.minInitialPoints) {
        reference_ = std::move(current);
        return;
    }

    // Match against the reference over the whole image: the motion between
    // the two frames is unknown and may be large.
    MatchParams matchParams = options_.match;
    matchParams.searchRadius = 0;
    std::vector<Match> matches;
    matchFeatures(current.keypoints, current.descriptors, reference_.keypoints, reference_.descriptors,
                  matchParams, matches, nullptr, pool_.get());

    const long age = current.index - reference_.index;
    if (static_cast<int>(matches.size()) < mp.minInitialPoints || age > mp.maxReferenceAge) {
        // The view has moved on (or too much time passed): start over from here.
        if (matches.size() < 0.3 * reference_.keypoints.size() || age > mp.maxReferenceAge) {
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
    k1.descriptors = reference_.descriptors;
    k1.colors = reference_.colors;
    k1.pointIds.assign(k1.keypoints.size(), -1);
    k2.frameIndex = current.index;
    k2.pose = {second.R, second.t * s};
    k2.keypoints = current.keypoints;
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
    Keyframe kf;
    kf.frameIndex = static_cast<long>(frameCount_);
    kf.pose = pose_;
    kf.keypoints = keypoints_;
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

    cullRecentPoints();

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
    for (int k : partners) triangulateBetween(k, n - 1);
    searchInNeighbors(n - 1);

    if (mp.localBAKeyframes > 0) {
        // The new keyframe and the keyframes sharing the most points with it
        // (counted after triangulation, which adds shared points).
        std::vector<int> local{n - 1};
        for (const auto& [k, shared] : covisibleKeyframes(n - 1)) {
            if (static_cast<int>(local.size()) >= mp.localBAKeyframes) break;
            local.push_back(k);
        }
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

BAReport Reconstructor::adjust(const std::vector<int>& keyframes, int iterations) {
    const int nk = static_cast<int>(keyframes_.size());
    if (nk < 2 || keyframes.empty()) return {};
    std::vector<char> variable(static_cast<std::size_t>(nk), 0);
    for (int k : keyframes) variable[k] = 1;

    // Points seen by the keyframes being adjusted.
    std::vector<int> pointIndex(points_.size(), -1);
    std::vector<int> pointIds;
    for (int k : keyframes) {
        for (int id : keyframes_[k].pointIds) {
            if (id >= 0 && !points_[id].bad && pointIndex[id] < 0) {
                pointIndex[id] = static_cast<int>(pointIds.size());
                pointIds.push_back(id);
            }
        }
    }
    if (pointIds.empty()) return {};

    // Cameras: every keyframe that sees those points (the adjusted ones plus
    // fixed neighbours), in keyframe order so the reduced system stays banded.
    std::vector<char> involved(static_cast<std::size_t>(nk), 0);
    for (int id : pointIds) {
        for (const PointObservation& o : points_[id].observations) involved[o.keyframe] = 1;
    }
    BAProblem problem;
    std::vector<int> cameraKeyframe;
    struct ObsRef {
        int keyframe, keypoint;
    };
    std::vector<ObsRef> obsRef;
    for (int k = 0; k < nk; ++k) {
        if (!involved[k]) continue;
        const Keyframe& kf = keyframes_[k];
        const int camera = static_cast<int>(cameraKeyframe.size());
        for (std::size_t i = 0; i < kf.pointIds.size(); ++i) {
            const int id = kf.pointIds[i];
            if (id < 0 || points_[id].bad || pointIndex[id] < 0) continue;
            problem.observations.push_back({camera, pointIndex[id],
                                            intrinsics_.normalize(kf.keypoints[i].x, kf.keypoints[i].y),
                                            kf.keypoints[i].scale});
            obsRef.push_back({k, static_cast<int>(i)});
        }
        cameraKeyframe.push_back(k);
        problem.poses.push_back(kf.pose);
        problem.fixedPose.push_back(variable[k] ? 0 : 1);
    }
    // Gauge: monocular maps are defined up to position, rotation and scale.
    // Without fixed neighbours, fix the two oldest keyframes.
    if (std::count(problem.fixedPose.begin(), problem.fixedPose.end(), 1) == 0) {
        for (std::size_t c = 0; c < problem.fixedPose.size() && c < 2; ++c) problem.fixedPose[c] = 1;
    }
    for (int id : pointIds) problem.points.push_back(points_[id].position);

    BAParams params;
    params.maxIterations = iterations;
    params.outlierThresholdPx = options_.mapping.maxReprojectionErrorPx + 0.5;
    const BAReport report = bundleAdjust(problem, intrinsics_.fx, params, pool_.get());
    if (!report.ok) return report;

    for (std::size_t c = 0; c < cameraKeyframe.size(); ++c) {
        if (!problem.fixedPose[c]) keyframes_[cameraKeyframe[c]].pose = problem.poses[c];
    }
    for (std::size_t j = 0; j < pointIds.size(); ++j) points_[pointIds[j]].position = problem.points[j];
    for (std::size_t o = 0; o < obsRef.size(); ++o) {
        if (report.inlier[o]) continue;
        const int id = keyframes_[obsRef[o].keyframe].pointIds[obsRef[o].keypoint];
        if (id >= 0) removeObservation(id, {obsRef[o].keyframe, obsRef[o].keypoint});
    }
    return report;
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

void Reconstructor::fuseIntoKeyframe(int k, const std::vector<int>& pointIds) {
    const MappingParams& mp = options_.mapping;
    const Keyframe& kf = keyframes_[k];
    const Vec3 center = kf.pose.center();
    const double cosMaxView = std::cos(mp.maxViewAngleDeg / kRadToDeg);
    const float scaleFactor = options_.orb.scaleFactor;
    const KeypointGrid grid(kf.keypoints, 16.0f);

    // Best keypoint for each point (read-only, in parallel), applied in order.
    std::vector<int> bestKeypoint(pointIds.size(), -1);
    parallelFor(pool_.get(), 0, static_cast<int>(pointIds.size()), 64, [&](int j0, int j1) {
        for (int j = j0; j < j1; ++j) {
            const MapPoint& p = points_[pointIds[j]];
            if (p.bad) continue;
            const Vec3 ray = p.position - center;
            const double dist = norm(ray);
            if (dist <= 0) continue;

            // Mean viewing direction, and the pyramid level the point should
            // appear at from here: a feature seen at level L from distance d0
            // appears about log(d0 / d) / log(scaleFactor) levels higher from d
            // (predicted from the observation at the most similar distance).
            bool seen = false;
            Vec3 meanDir;
            double bestLogRatio = 1e30;
            int predictedLevel = p.level;
            for (const PointObservation& o : p.observations) {
                if (o.keyframe == k) seen = true;
                const Vec3 d0 = p.position - keyframes_[o.keyframe].pose.center();
                meanDir = meanDir + normalized(d0);
                const double logRatio = std::log(norm(d0) / dist);
                if (std::abs(logRatio) < std::abs(bestLogRatio)) {
                    bestLogRatio = logRatio;
                    const int level = keyframes_[o.keyframe].keypoints[o.keypoint].level;
                    predictedLevel = static_cast<int>(std::lround(level + logRatio / std::log(scaleFactor)));
                }
            }
            if (seen || dot(ray, normalized(meanDir)) < cosMaxView * dist) continue;
            predictedLevel = std::clamp(predictedLevel, 0, std::max(options_.orb.levels - 1, 0));

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

    for (std::size_t j = 0; j < pointIds.size(); ++j) {
        const int kp = bestKeypoint[j];
        int id = pointIds[j];
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
    for (int t : targets) {
        std::vector<int> ids;
        for (int id : keyframes_[k].pointIds) {
            if (id >= 0 && !points_[id].bad) ids.push_back(id);
        }
        fuseIntoKeyframe(t, ids);
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
    fuseIntoKeyframe(k, ids);
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
    const int iterations = options_.mapping.globalBAIterations;
    if (iterations <= 0 || keyframes_.size() < 2) return;
    auto t = profiler_.scope("global ba");
    std::vector<int> all(keyframes_.size());
    for (std::size_t k = 0; k < all.size(); ++k) all[k] = static_cast<int>(k);
    stats_.globalBA = adjust(all, iterations);
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
            for (std::size_t a = 0; a < ia.size(); ++a) {
                const Keypoint& ka = older.keypoints[ia[a]];
                if (std::abs(ka.level - kb.level) > maxLevelDiff) continue;
                const double lineDistPx = std::abs(line.x * xa[a].x + line.y * xa[a].y + line.z) * toPx;
                if (lineDistPx > mp.epipolarDistancePx * ka.scale) continue;
                const int d = hammingDistance(db, older.descriptors[ia[a]]);
                if (d < bestDist) {
                    second = bestDist;
                    bestDist = d;
                    bestA = static_cast<int>(a);
                } else if (d < second) {
                    second = d;
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
    reference_ = FrameData{};
    state_ = State::Initializing;
    hasVelocity_ = false;
    lostCount_ = 0;
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
