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

    for (std::size_t i = 0; i < matches.size(); ++i) {
        if (!good[i]) continue;
        MapPoint p;
        p.position = X[i] * s;
        p.descriptor = k2.descriptors[matches[i].query];
        p.level = k2.keypoints[matches[i].query].level;
        p.color = k1.colors[matches[i].train];
        p.observations = 2;
        const int id = static_cast<int>(points_.size());
        points_.push_back(p);
        k1.pointIds[matches[i].train] = id;
        k2.pointIds[matches[i].query] = id;
    }

    pose_ = k2.pose;
    previousPose_ = k2.pose;
    hasVelocity_ = false;  // the pair may span many frames; no per-frame velocity yet
    trackedPoints_ = k2.pointIds;
    trackedCount_ = static_cast<int>(depths.size());
    keyframes_.push_back(std::move(k1));
    keyframes_.push_back(std::move(k2));
    lastKeyframeFrame_ = current.index;
    lastWasKeyframe_ = true;
    reference_ = FrameData{};
    state_ = State::Tracking;
    lostCount_ = 0;
    ++stats_.maps;
}

std::vector<int> Reconstructor::localPoints() const {
    std::vector<char> seen(points_.size(), 0);
    std::vector<int> ids;
    const int first = std::max(0, static_cast<int>(keyframes_.size()) - options_.mapping.localKeyframes);
    for (int k = static_cast<int>(keyframes_.size()) - 1; k >= first; --k) {
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
    kf.pointIds = trackedPoints_;
    for (std::size_t k = 0; k < kf.pointIds.size(); ++k) {
        const int id = kf.pointIds[k];
        if (id < 0) continue;
        MapPoint& p = points_[id];
        ++p.observations;
        p.descriptor = kf.descriptors[k];  // keep the newest view's appearance
        p.level = kf.keypoints[k].level;
    }
    keyframes_.push_back(std::move(kf));
    lastKeyframeFrame_ = static_cast<long>(frameCount_);
    lastWasKeyframe_ = true;

    // Triangulation partners: the most recent keyframes whose baseline relative
    // to the scene depth gives enough parallax (newest first, as collected).
    const MappingParams& mp = options_.mapping;
    const int n = static_cast<int>(keyframes_.size());
    const Keyframe& newest = keyframes_[n - 1];
    const double depth = medianSceneDepth(newest);
    std::vector<std::pair<double, int>> partners;  // (baseline ratio, keyframe index)
    for (int k = n - 2; k >= 0 && k >= n - 1 - mp.triangulationSearch; --k) {
        const double baseline = norm(keyframes_[k].pose.center() - newest.pose.center());
        const double ratio = depth > 0 ? baseline / depth : 0.0;
        if (ratio >= mp.minBaselineRatio) partners.push_back({ratio, k});
    }
    // Most recent qualifying keyframes first: enough parallax, most shared view.
    if (static_cast<int>(partners.size()) > mp.triangulationKeyframes) partners.resize(mp.triangulationKeyframes);
    for (const auto& [ratio, k] : partners) triangulateBetween(keyframes_[k], keyframes_[n - 1]);

    if (mp.localBAKeyframes > 0) {
        const BAReport r = adjust(std::max(0, n - mp.localBAKeyframes), mp.localBAIterations);
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

BAReport Reconstructor::adjust(int firstKeyframe, int iterations) {
    const int nk = static_cast<int>(keyframes_.size());
    if (nk < 2 || firstKeyframe >= nk) return {};

    // Points seen by the keyframes being adjusted.
    std::vector<int> pointIndex(points_.size(), -1);
    std::vector<int> pointIds;
    for (int k = firstKeyframe; k < nk; ++k) {
        for (int id : keyframes_[k].pointIds) {
            if (id >= 0 && !points_[id].bad && pointIndex[id] < 0) {
                pointIndex[id] = static_cast<int>(pointIds.size());
                pointIds.push_back(id);
            }
        }
    }
    if (pointIds.empty()) return {};

    // Cameras: the adjusted keyframes plus fixed ones that also see those points.
    BAProblem problem;
    std::vector<int> cameraKeyframe;
    struct ObsRef {
        int keyframe, keypoint;
    };
    std::vector<ObsRef> obsRef;
    for (int k = 0; k < nk; ++k) {
        const bool variable = k >= firstKeyframe;
        const Keyframe& kf = keyframes_[k];
        const int camera = static_cast<int>(cameraKeyframe.size());
        bool used = false;
        for (std::size_t i = 0; i < kf.pointIds.size(); ++i) {
            const int id = kf.pointIds[i];
            if (id < 0 || points_[id].bad || pointIndex[id] < 0) continue;
            problem.observations.push_back(
                {camera, pointIndex[id], intrinsics_.normalize(kf.keypoints[i].x, kf.keypoints[i].y)});
            obsRef.push_back({k, static_cast<int>(i)});
            used = true;
        }
        if (!used) continue;
        cameraKeyframe.push_back(k);
        problem.poses.push_back(kf.pose);
        problem.fixedPose.push_back(variable ? 0 : 1);
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
        Keyframe& kf = keyframes_[obsRef[o].keyframe];
        const int id = kf.pointIds[obsRef[o].keypoint];
        kf.pointIds[obsRef[o].keypoint] = -1;
        if (--points_[id].observations < 2) points_[id].bad = true;
    }
    return report;
}

void Reconstructor::finish() {
    const int iterations = options_.mapping.globalBAIterations;
    if (iterations <= 0 || keyframes_.size() < 2) return;
    auto t = profiler_.scope("global ba");
    stats_.globalBA = adjust(0, iterations);
    stats_.globalBACameras = static_cast<int>(keyframes_.size());
    stats_.globalBAPoints = static_cast<int>(std::count_if(points_.begin(), points_.end(),
                                                           [](const MapPoint& p) { return !p.bad; }));
}

int Reconstructor::triangulateBetween(Keyframe& older, Keyframe& newer) {
    const MappingParams& mp = options_.mapping;
    const double fx = intrinsics_.fx;

    // Features not yet attached to map points.
    std::vector<int> ia, ib;
    std::vector<Keypoint> ka, kb;
    std::vector<Descriptor> da, db;
    for (std::size_t i = 0; i < older.pointIds.size(); ++i) {
        if (older.pointIds[i] < 0) {
            ia.push_back(static_cast<int>(i));
            ka.push_back(older.keypoints[i]);
            da.push_back(older.descriptors[i]);
        }
    }
    for (std::size_t i = 0; i < newer.pointIds.size(); ++i) {
        if (newer.pointIds[i] < 0) {
            ib.push_back(static_cast<int>(i));
            kb.push_back(newer.keypoints[i]);
            db.push_back(newer.descriptors[i]);
        }
    }
    MatchParams matchParams = options_.match;
    matchParams.searchRadius = 0;  // keyframes can be far apart in any direction
    std::vector<Match> matches;
    matchFeatures(kb, db, ka, da, matchParams, matches, nullptr, pool_.get());

    // Relative pose older -> newer is known: check epipolar consistency, then
    // triangulate in the older camera frame.
    const Pose rel = older.pose.inverse().then(newer.pose);
    const Mat3 E = skew(rel.t) * rel.R;
    const double maxSampson2 = std::pow(mp.maxReprojectionErrorPx / fx, 2);
    const Vec3 c2 = rel.center();
    const Pose olderInv = older.pose.inverse();

    int created = 0;
    for (const Match& m : matches) {
        const int a = ia[m.train], b = ib[m.query];
        const Vec2 x1 = intrinsics_.normalize(older.keypoints[a].x, older.keypoints[a].y);
        const Vec2 x2 = intrinsics_.normalize(newer.keypoints[b].x, newer.keypoints[b].y);
        if (!(sampsonError(E, x1, x2) < maxSampson2)) continue;  // also rejects pure rotation (E ~ 0)
        Vec3 X;
        if (!triangulate(x1, x2, rel.R, rel.t, X)) continue;
        if (X.z <= 0 || rel.transform(X).z <= 0) continue;
        if (triangulationAngleDeg(X, {}, c2) < mp.minTriangulationAngleDeg) continue;
        if (reprojectionErrorPx(Pose{}, X, x1, fx) > mp.maxReprojectionErrorPx) continue;
        if (reprojectionErrorPx(rel, X, x2, fx) > mp.maxReprojectionErrorPx) continue;

        MapPoint p;
        p.position = olderInv.transform(X);
        p.descriptor = newer.descriptors[b];
        p.level = newer.keypoints[b].level;
        p.color = newer.colors[b];
        p.observations = 2;
        const int id = static_cast<int>(points_.size());
        points_.push_back(p);
        older.pointIds[a] = id;
        newer.pointIds[b] = id;
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
