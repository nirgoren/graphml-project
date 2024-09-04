# Look at super resolution for images with diffusion
## Training
Add Gaussian noise to all points at each iteration (no need to go all the way to random i.i.d noise since in test time we will start from only a slightly noised point cloud). Keep original graph based on the KNN graph of the original point cloud.  Embed diffustion timestamp feature in node features.

Goal - predict noise (L2 loss).

## Test time
Add points around the sparse points with some Gaussian noise. Build graph based on KNN and iteratively denoise.
## Network arch
GCN? Use distance features on the edges? Use different node types (heterogenious graph) i.e marking the un-noised points?