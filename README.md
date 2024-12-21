# Image Processing and Computer Vision Assignments

## Assignment Module 1: **Product Recognition of Food Products**

**Goal**: Develop a computer vision system that, given a reference image for each product, is able to identify such product from one picture of a store shelf.

**Implementation Details**: We tackle this assignment by implementing from scratch a `Generalized Hough Transform (GHT)` with Local Invariant Features. To improve accuracy, we integrated a color consistency check on the detected bounding boxes to filter out incorrect matches. This improvement is crucial because the GHT processes only greyscale images and cannot differentiate between templates that are visually identical in shape but differ in color.

The provided scene images were significantly ruined by **salt noise**. After conducting a noise analysis, we identified that applying a combination of `Median Filtering, BM3D, Non-Local Means Filtering, and sharpening` significantly increased the number of keypoints detected in the scene images. This improvement greatly enhanced the performance of our algorithm.

### Example Images:

<p align="center"> 
  <img src="./media/star_models.png" alt="Reference product templates used for recognition" width="100%" /> 
  <br>
  <em>Figure 1: Reference product templates used for recognition.</em> 
</p> 
<p align="center"> 
  <img src="./media/products_recognition.png" alt="Recognition results on a store shelf image" width="100%" /> 
  <br>
  <em>Figure 2: Recognition results on a store shelf image.</em> 
</p> 
<p align="center"> 
  <img src="./media/noise_analysis.png" alt="Noise analysis results with keypoint improvements" width="100%" />
  <br>
  <em>Figure 3: Visualization of noise reduction.</em> 
</p>

---

## Assignment Module 2: **Product Classification**

**Goal**: Implement a neural network that classifies smartphone pictures of products found in grocery stores.

**Implementation Details**: To facilitate neural network training on our laptops, we explored efficient architectures, focusing particularly on `ShuffleNet`. This architecture builds upon the **depthwise separable convolution** introduced in Xception by incorporating **grouped pointwise convolutions** and a novel **shuffle layer**. Leveraging these shuffle units, we implemented a compact neural network that achieved 72% accuracy on the validation set without any prior knowledge or pretraining. After we use a pretrained net (`Resnet-18`) achieving 91% accuracy also on validation set, showing the impact of transfer learning and prior knowledge in improving model performance. 

### Example Images:

<p align="center">
  <img src="https://github.com/marcusklasson/GroceryStoreDataset/raw/master/sample_images/natural/Granny-Smith.jpg" width="150" alt="Granny Smith">
  <img src="https://github.com/marcusklasson/GroceryStoreDataset/raw/master/sample_images/natural/Pink-Lady.jpg" width="150" alt="Pink Lady">
  <img src="https://github.com/marcusklasson/GroceryStoreDataset/raw/master/sample_images/natural/Lemon.jpg" width="150" alt="Lemon">
  <img src="https://github.com/marcusklasson/GroceryStoreDataset/raw/master/sample_images/natural/Banana.jpg" width="150" alt="Banana">
  <img src="https://github.com/marcusklasson/GroceryStoreDataset/raw/master/sample_images/natural/Vine-Tomato.jpg" width="150" alt="Vine Tomato">
</p>

<p align="center">
  <img src="https://github.com/marcusklasson/GroceryStoreDataset/raw/master/sample_images/natural/Yellow-Onion.jpg" width="150" alt="Yellow Onion">
  <img src="https://github.com/marcusklasson/GroceryStoreDataset/raw/master/sample_images/natural/Green-Bell-Pepper.jpg" width="150" alt="Green Bell Pepper">
  <img src="https://github.com/marcusklasson/GroceryStoreDataset/raw/master/sample_images/natural/Arla-Standard-Milk.jpg" width="150" alt="Arla Standard Milk">
  <img src="https://github.com/marcusklasson/GroceryStoreDataset/raw/master/sample_images/natural/Oatly-Natural-Oatghurt.jpg" width="150" alt="Oatly Natural Oatghurt">
  <img src="https://github.com/marcusklasson/GroceryStoreDataset/raw/master/sample_images/natural/Alpro-Fresh-Soy-Milk.jpg" width="150" alt="Alpro Fresh Soy Milk">
</p>
