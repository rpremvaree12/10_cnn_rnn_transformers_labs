# Lab 1: CNN-Feature Hierarchy and Receptive Fields

**Explain how CNNs build a hierarchy of features through their layer structure. In your answer:

* Describe what types of features are typically detected at early, middle, and deep layers of a CNN
* Explain the concept of a receptive field and how it changes as we move deeper into the network
* Illustrate with a specific example of how this hierarchical feature extraction might work in developing a a Convolutional Neural Network (CNN) approach, as CNNs excel at extracting meaningful visual features from images.

## How CNNs Build a Feature Hierarchy

A Convolutional Neural Network processes images by repeatedly applying a convolution operation, in which a small grid of learned weights called a kernel slides across the input and computes a dot product at each position. The result of this computation is a feature map — a spatial record of where and how strongly a particular pattern was detected across the image. By stacking many of these convolution layers on top of each other, a CNN builds an increasingly abstract understanding of the image, with each layer learning to detect more complex features than the one before it. This progression from simple to complex is what we mean by a feature hierarchy.

**Early layers** detect low-level, local patterns that are essentially universal building blocks of any image. These include edges, color gradients, corners, and simple textures. A kernel in the first layer might learn to fire strongly whenever it encounters a sharp contrast between two neighboring pixel values — which is the fundamental signature of an edge. These features are small, local, and not yet meaningful on their own, but they form the foundation for everything that follows.

**Middle layers** combine those early features into increasingly structured patterns. Rather than detecting a single edge, a middle layer neuron might respond to a combination of edges arranged in a particular orientation — a scratch line, a curved boundary, a repeating surface texture. At this stage the network is starting to detect things that begin to resemble the kinds of visual structures a human inspector might notice, though still not a complete defect.

**Deep layers** near the end of the network detect high-level, semantically meaningful features. By this point each neuron is responding to complex combinations of everything the earlier layers detected. In the context of our parts dataset, a deep layer neuron might fire strongly on the full pattern characteristic of a crack — irregular edges, abnormal surface texture, and a particular shape — while remaining silent on the smooth, uniform appearance of a non-defective part. The final classification layer then takes these deep representations and produces the output: defective or non-defective.

This mirrors what a human quality inspector does naturally. They don't consciously process raw pixels — they first notice low-level irregularities, then recognize them as a meaningful defect pattern. The CNN automates this process, and critically, it learns *which* features actually matter by training on our 50,000 labeled images rather than relying on hand-crafted rules.

## Receptive Fields and How They Grow

To understand why deep layers can detect whole-part defect patterns while early layers can only see local scratches, we need to understand the concept of a receptive field. A neuron's receptive field is the region of the original input image that contributed to that neuron's value — essentially, how much of the image that neuron "sees."

In the first convolution layer, each neuron has a small receptive field determined directly by the kernel size. With a 3×3 kernel, each neuron is only looking at a 3×3 patch of the original image. This is why early layers can only detect small, local features.

As we move deeper, the receptive field grows — not because the kernels get larger, but because each neuron in a deeper layer is responding to neurons in the previous layer that were themselves already responding to different regions of the image. A neuron in layer 2 might look at a 3×3 patch of layer 1's feature map, but each of those layer 1 neurons was already looking at a 3×3 patch of the original image. The layer 2 neuron is therefore informed by a 5×5 region of the original image. Stack enough layers and a single neuron in a deep layer may have a receptive field spanning most or all of the input image.

This growth in receptive field size is the mechanism that allows deep layers to understand large-scale structure. A crack spanning a significant portion of a part's surface cannot be detected by a neuron with a tiny receptive field — it requires a neuron that can integrate information from a wide region of the image. The hierarchical layer structure of a CNN naturally produces this capability through the accumulation of receptive fields across layers.

## Applying This to the Parts Defect Detection Task

With 50,000 labeled images of defective and non-defective parts, we can trace exactly how this feature hierarchy would work in practice.

In the earliest layers, the network would learn kernels that detect the basic visual building blocks present in images of manufactured parts — the sharp edges of component boundaries, the smooth gradients of a clean metal surface, and the subtle texture differences between normal and abnormal material. At this stage the network has no concept of "defect" — it is simply learning to see.

In the middle layers, those basic features get combined into more structured patterns. The network might begin to detect elongated scratch-like structures by combining edge detectors oriented in the same direction, or identify regions of irregular texture by combining responses from multiple texture-sensitive kernels. These intermediate features are more specific to the domain of manufactured parts, and less likely to resemble what a network trained on photographs of natural scenes would develop.

In the deep layers, the network assembles those mid-level patterns into the full signature of a defect. A crack has a characteristic combination of properties — it produces edges in an irregular, branching pattern, disrupts the surface texture, and typically has a particular aspect ratio and orientation. A deep layer neuron that learned to respond to this combination would be an effective crack detector. Non-defective parts, by contrast, would produce a different deep layer activation pattern characterized by smooth, regular, predictable features throughout.

It is also worth noting that for this task, we would likely benefit from transfer learning — beginning with a model like ResNet or VGG that was pre-trained on a large general image dataset. The early and middle layer kernels learned from millions of natural images are already good general-purpose feature detectors. Rather than learning from scratch how to detect edges and textures, we can inherit those capabilities and focus the training process on the deeper layers that need to learn the specific patterns that distinguish defective from non-defective parts. Given that 50,000 images is a reasonable but not enormous dataset, this approach would help the network learn more robust deep features than training from random kernel values alone.

In summary, the power of a CNN for this defect detection task lies precisely in this hierarchical structure — the ability to automatically discover, through training, the chain of features from raw pixel edges all the way up to meaningful defect patterns, guided entirely by the labeled examples in our dataset.
