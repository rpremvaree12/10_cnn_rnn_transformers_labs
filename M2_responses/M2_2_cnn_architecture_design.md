# Lab 2: CNN Architecture Design Decisions

Discuss the following key architectural decisions you would make and why:

1. What kernel sizes would you choose for different layers and why?
2. How would you incorporate pooling layers and what benefits would they provide in this specific application?
3. What considerations would guide your decisions about network depth (number of layers) and width (number of filters per layer)?
4. How might you leverage transfer learning in this scenario, and what would be the advantages?

## 1. Kernel Sizes Across Layers

Kernel size determines how large a patch of the input each neuron examines at a given layer, and the choice involves a tradeoff between capturing local detail and computational efficiency.

For the early layers of our defect detection network, smaller kernels — specifically 3×3 — are the right choice. Defects like cracks and surface scratches are defined by sharp, local discontinuities in texture and edge patterns. A 3×3 kernel is precisely sized to detect these kinds of small, high-frequency signals without averaging over too large a region and losing that fine-grained detail. There is also a well-established finding in the deep learning literature, demonstrated prominently by the VGG architecture, that stacking multiple 3×3 convolutions achieves the same effective receptive field as a single larger kernel while using fewer parameters and introducing more non-linearity — both of which tend to improve learning.

In middle layers, where the network is combining local features into more structured patterns, 3×3 kernels remain appropriate for the same reasons. The receptive field has grown naturally through layer stacking, so there is no need to increase kernel size to capture larger structures — the depth of the network handles that.

For an initial entry point into the image, some architectures use a single larger kernel — 5×5 or 7×7 — in the very first layer. This can be useful when the input images are high resolution, as it allows the first layer to quickly summarize large regions before passing to the stack of 3×3 layers. Given that our dataset consists of production line images that are likely high resolution and detail-rich, a 5×5 or 7×7 first layer kernel followed by 3×3 kernels throughout the rest of the network would be a reasonable design choice.

## 2. Pooling Layers and Their Role

Pooling layers reduce the spatial dimensions of feature maps between convolution layers, and they serve several important purposes in a defect detection context.

The most common choice is **max pooling**, typically with a 2×2 window and a stride of 2, which halves the height and width of the feature map while retaining the strongest activation in each region. For defect detection this is particularly well suited — we care about whether a defect feature is present somewhere in a region, not precisely where within that region it falls. Max pooling preserves the "something was detected here" signal while discarding exact positional information that isn't necessary for a binary defective/non-defective classification.

Pooling also provides **translation invariance** — the network becomes less sensitive to small shifts in where a defect appears in the image. A crack in the upper-left quadrant of a part and a crack in the lower-right quadrant should both be classified as defective. Pooling helps the network learn this without needing to see every possible defect location during training.

From a practical standpoint, pooling reduces the number of values flowing through the network at each stage, which decreases memory requirements and computation time. With 50,000 images this is manageable, but pooling still speeds up training meaningfully and reduces the risk of overfitting by limiting the total number of parameters the network needs to learn.

A reasonable pooling strategy for this network would be to insert a 2×2 max pooling layer after every two or three convolution layers — enough to progressively reduce spatial dimensions without discarding spatial information too aggressively in the early stages where fine defect detail still matters.

## 3. Network Depth and Width

Depth refers to the number of layers and width refers to the number of filters per layer — together they determine the total capacity of the network, meaning how complex a function it can learn.

**On depth:** The right number of layers depends on how complex the patterns we need to detect are. Defect detection is a moderately complex visual task — more complex than detecting whether an image is bright or dark, but less complex than full object recognition across hundreds of categories. A network in the range of 6 to 12 convolution layers is likely appropriate. Too few layers and the network won't develop the deep representations needed to distinguish subtle defect patterns from normal surface variation. Too many layers and the network becomes harder to train, more prone to overfitting on our 50,000 image dataset, and more computationally expensive without meaningful gains in accuracy. If we are using transfer learning from a pre-trained model like ResNet, the depth decision is largely made for us — we inherit the pre-trained depth and fine-tune from there.

**On width:** The number of filters per layer determines how many distinct features the network can detect at each stage. A common and sensible pattern is to start with fewer filters in early layers and progressively increase the count in deeper layers — for example, 32 filters in the first layer, 64 in the next, 128 in the middle layers, and so on. The reasoning is that early layers only need to detect a relatively small vocabulary of low-level features like edges and textures, while deeper layers need to represent a much richer set of combinations. Starting narrow and widening gradually balances expressiveness against computational cost.

For our defect detection task specifically, width also needs to account for the variety of defect types in the dataset. If the 50,000 images contain multiple distinct defect categories — cracks, dents, discoloration, missing components — the network needs enough filters in its deep layers to represent each of those patterns distinctly. If the task is strictly binary (defective vs. non-defective regardless of defect type), a narrower network may suffice.

**The overfitting consideration:** With 50,000 images, overfitting is a genuine risk if the network is too deep or too wide. Every additional layer and filter adds parameters that need to be learned from the available data. Regularization techniques like dropout and batch normalization can help, but the most robust solution is to keep the network only as large as the task demands — which for binary defect classification on manufacturing parts is probably more modest than the large architectures used for ImageNet-scale problems.

## 4. Transfer Learning

Transfer learning is almost certainly the right approach for this task, and it would be the first architectural decision I would make before settling on depth, width, or kernel sizes.

The core idea is that a model pre-trained on a large general image dataset — ImageNet is the standard choice, with over a million labeled images across a thousand categories — has already learned a rich set of general-purpose visual features in its early and middle layers. Edges, textures, curves, surface gradients — all of the low-level building blocks that our defect detection network would need to learn from scratch are already encoded in those pre-trained kernels. There is no reason to re-learn them from our 50,000 images when we can inherit them from a model that learned them from millions.

A strong starting point would be **ResNet50** or **EfficientNet**, both of which are available pre-trained in Keras and PyTorch. These architectures have proven effective across a wide range of visual tasks and are well-documented. The general strategy would be to load the pre-trained model, remove its final classification layer (which was designed for 1,000 ImageNet categories), and replace it with a new classification head suited to our binary defect detection output.

From there, there are two approaches depending on how similar our parts images are to the ImageNet training data. If the images are fairly typical — well-lit, in focus, showing recognizable surfaces — we might **freeze** the early layers entirely and only train the new classification head plus the deepest few layers on our dataset. This is faster, requires less data, and avoids corrupting the well-learned general features with noisy updates from a smaller dataset. If our images are quite different from natural photographs — unusual lighting, microscopic scale, highly specialized textures — we might **fine-tune** the entire network, allowing all layers to adjust, but starting from the pre-trained weights rather than random initialization.

The practical advantages for this project are significant. Training time is dramatically reduced because we are not learning from random kernel values. The network reaches good performance with less data, which matters because 50,000 images is a reasonable dataset but not a large one by deep learning standards. And the risk of the network failing to learn meaningful features at all — a real danger when training from scratch with limited data — is substantially mitigated because the foundational visual vocabulary is already in place before training begins.

In summary, the architectural decisions are not independent — they interact. Transfer learning shapes the depth and width decisions because we inherit an existing architecture. Kernel size choices are guided by the nature of defects we are trying to detect. Pooling strategy reflects what spatial information matters for classification. And network capacity must be balanced against the size of our dataset to avoid overfitting. Thinking through these decisions together, before writing any code, is what makes the difference between a network that learns and one that doesn't.
