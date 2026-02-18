# Lab 3: Comparing CNNs to Traditional Neural Networks

A team member who is new to deep learning has asked you to explain why CNNs are preferred over fully-connected neural networks for image classification tasks.

Write a comprehensive explanation that:

* Compares parameter efficiency between CNNs and fully-connected networks using a specific image size example
* Explains how parameter sharing and local connectivity in CNNs address the limitations of fully-connected networks
* Describes how CNNs maintain spatial information that would be lost in traditional neural networks
* Illustrates a specific problem where a fully-connected network would struggle but a CNN would excel, with reasoning.

## The Core Problem with Fully-Connected Networks on Images

To understand why CNNs are preferred for image tasks, it helps to first understand what a fully-connected network — sometimes called a dense network or multilayer perceptron — actually does with an image. In a fully-connected network, every neuron in one layer is connected to every neuron in the next layer. Every single input value influences every single neuron in the first hidden layer. This sounds thorough, but for images it creates two serious problems: an unmanageable number of parameters, and a complete disregard for the spatial structure of the image.

## Parameter Efficiency: A Concrete Comparison

Consider a modest grayscale image of 128×128 pixels. That image contains 16,384 pixel values. In a fully-connected network, if the first hidden layer has 512 neurons, then every one of those 512 neurons needs a separate weight connecting it to every one of the 16,384 input pixels. That's 512 × 16,384 = **8,388,608 parameters** in the first layer alone, before the network has learned anything meaningful.

Now consider that real images are rarely grayscale or that small. A color image at 224×224 pixels — the standard input size for many pre-trained models — has 224 × 224 × 3 = **150,528 input values**. A fully-connected first layer with 512 neurons would require over 77 million parameters just in that first layer. A network with multiple such layers would have hundreds of millions of parameters, all of which need to be learned from training data. This creates enormous memory requirements, extremely slow training, and a very high risk of overfitting because the network has vastly more parameters than it has meaningful patterns to learn.

A CNN processing the same 224×224×3 image with a first layer of 32 filters, each using a 3×3 kernel, requires only 3 × 3 × 3 × 32 = **864 parameters** in that first layer. The difference is not marginal — it is a difference of several orders of magnitude. This is possible because of two fundamental design principles: local connectivity and parameter sharing.

## Local Connectivity

In a fully-connected network, each neuron connects to every input. In a CNN, each neuron connects only to a small local region of the input — the patch covered by its kernel. This is called local connectivity, and it is justified by a simple observation about images: pixels that are spatially close to each other are far more likely to be related than pixels that are far apart. The pixels that make up a crack in a part are neighboring pixels. A neuron that only examines a local neighborhood misses nothing important about local structure, while a neuron connected to the entire image is wasting most of its connections on pixel relationships that carry no useful information.

Local connectivity alone dramatically reduces the number of parameters. But the even larger savings comes from what happens next.

## Parameter Sharing

In a fully-connected network, every neuron has its own unique set of weights. Neuron 1 and Neuron 2 in the first hidden layer have completely independent parameters, even if they are detecting the same kind of feature in different parts of the image.

In a CNN, every neuron in a given feature map shares the same kernel weights. The same 3×3 edge-detecting kernel that slides across the top-left corner of the image is the identical kernel that slides across the bottom-right corner. This is parameter sharing, and it rests on a crucial assumption: a feature that is meaningful in one part of the image is meaningful everywhere in the image. An edge is an edge regardless of where it appears. A crack texture is a crack texture whether it is in the center or the corner of the part.

Parameter sharing means that instead of learning a separate edge detector for every location in the image, the network learns one edge detector and applies it everywhere. This is not a compromise — it is actually a stronger model, because it forces the network to learn features that generalize across locations rather than memorizing position-specific patterns that may not appear the same way in new images.

## Preserving Spatial Information

A fully-connected network requires its input to be a flat one-dimensional vector. To feed a 224×224×3 image into a fully-connected network, you must first unroll it into a vector of 150,528 values. In doing so, you destroy all spatial relationships. The pixel at position (10, 10) and the pixel at position (11, 10) — which are immediate neighbors — end up as entries 6,730 and 6,954 in the vector, with thousands of unrelated values between them. The network receives no information about which pixels were adjacent, which were part of the same region, or how any spatial structures were arranged.

A CNN never flattens the image until the very end. It processes the image as a two-dimensional grid throughout all of its convolution layers, preserving the neighborhood relationships between pixels at every stage. When a middle-layer neuron detects a scratch-like structure, it knows where in the image that structure was, and passes that spatial information forward. The feature maps produced by each convolution layer are themselves two-dimensional grids that maintain the spatial layout of detected features. This is why CNNs can detect not just that a defect is present, but roughly where in the image it is located — something a fully-connected network is structurally incapable of.

## A Specific Example: Detecting a Hairline Crack

Consider the problem of detecting a hairline crack in a manufactured part. The crack might appear anywhere on the part's surface — top, bottom, left, right, diagonal. It might be a few pixels wide and span a significant portion of the image. Its defining visual characteristic is a narrow, elongated region of sharp edge contrast cutting across an otherwise smooth surface.

A fully-connected network would struggle with this for several compounding reasons. First, because it sees the image as a flat vector, it has no concept of "elongated" or "edge" — those are inherently spatial descriptions. Second, because every neuron connects to every input independently, the network would need to separately memorize what a crack looks like in every possible position and orientation it might appear in the training data. If a crack appears in the upper-left in training but the upper-right in a test image, a fully-connected network may fail to recognize it. Third, the enormous number of parameters makes it very likely the network overfits to the specific cracks it saw during training rather than learning the general concept of a crack.

A CNN handles this naturally. Its early layers learn edge-detecting kernels that fire on the sharp contrast boundaries of the crack regardless of where they appear. Middle layers combine those edge responses into elongated structures. Deep layers recognize the full crack pattern. Because the same kernels scan the entire image through parameter sharing, the network detects the crack whether it appears in the upper-left or lower-right. Because spatial information is preserved throughout, the network can represent the elongated, directional nature of the crack rather than just detecting that some unusual pixel values exist somewhere in the flattened vector.

## Summary

The preference for CNNs over fully-connected networks for image tasks is not arbitrary — it follows directly from the structure of images themselves. Images have local structure, meaning nearby pixels are related. Images have translation invariance, meaning features mean the same thing regardless of where they appear. And images have spatial organization, meaning the arrangement of features carries information. Fully-connected networks ignore all three of these properties. CNNs are built around all three of them — through local connectivity, parameter sharing, and spatial preservation respectively. The result is a model that is simultaneously more efficient, more generalizable, and more capable on visual tasks than any fully-connected alternative of comparable size.
