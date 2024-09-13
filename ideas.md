# Look at super resolution for images with diffusion
## Taking inspiration from ResShift
1. Upsample the sparse point cloud by iteratively inserting a point on the longest edge of the nn graph (to get the LR point cloud).
2. During training: noise the ground truth (HR) point cloud towards the LR point cloud instance using normal based direction: For every point in high resolution, find nearest neighbor in LR and push along normal towards neighbor (+ add noise). concat LR xyz features as conditioning
3. Consider adding the normal direction as additional features
4. Inference - upsample, add noise and run diffusion network
