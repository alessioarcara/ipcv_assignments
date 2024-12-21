# Image Processing and Computer Vision Assignments

## Assignment Module 1: **Product Recognition of Food Products**

**Goal**: Develop a computer vision system that, given a reference image for each product, is able to identify such product from one picture of a store shelf.

**Implementation Details**: We tackle this assignment by implementing from scratch a `Generalized Hough Transform (GHT)` with Local Invariant Features. To improve accuracy, we integrated a color consistency check on the detected bounding boxes to filter out incorrect matches. This improvement is crucial because the GHT processes only greyscale images and cannot differentiate between templates that are visually identical in shape but differ in color.

The provided scene images were significantly ruined by **salt noise**. After conducting a noise analysis, we identified that applying a combination of `Median Filtering, BM3D, Non-Local Means Filtering, and sharpening` significantly increased the number of keypoints detected in the scene images. This improvement greatly enhanced the performance of our algorithm.

### Example Images:

<p align="center">
  <img src="https://i.ibb.co/TwkMWnH/Screenshot-2024-04-04-at-14-54-51.png" alt="Store Shelf Example" width="300" />
</p>

---

## Assignment Module 2: **Product Classification**

**Goal**: Implement a neural network that classifies smartphone pictures of products found in grocery stores.

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
