# README

My code is all in the Yes.ipynb file. All requirements are listed in requirements.txt

## Questions

1. Why are you designing the solution in this way?

    The main constraint in my design was the time I was given as well as the fact that the data is fully unsupervised. Given the fact that I was working with unlabeled data I thought of three options.

    1. View this as a clustering problem. Do multimodal clustering on the text and images and label the clusters. Hope that the clusters align well with the desired classes. 
    2. Hand label a golden set and treat this as a supervised problem
    3. Use pretrained models to treat this as a zero-shot classification problem. 

    Option 1 to me felt like it probably wouldn't work, as the data is very high dimensional (img + text), and it seemed unlikely that unsupervised clustering would pick up on the class patterns that we care about, rather than some spurious patterns. 

    I strongly considered option 2, but due to the time constraint, was worried that I would not have enough time to implement a solution in addition to working through the hand labeling. If I had more time, I believe this option would have given the best results. It would have been very helpful to have a test set with which I could compare different options empirically that I know is correctly labeled. 

    I ended up going with option 3, of treating this like a zero-shot classification problem. At a high level, I utilized two pretrained models (CLIP + MiniLM) to embed the images and descriptions and used these embeddings to compute distances between the input and the classes. When looking through the data in my eda, I concluded that there are inputs that would be the most challenging would be the ones where the inputs contains multiple classes, in which case it is required to use both the image and text to disambiguate what class is the correct one.

    I also found that the classes were not really descriptive enough (for myself or for creating an embedding). So I by hand augmented the classes with some subclasses (tops -> shirt, t-shirt, blouse, etc. (you can see these in the ipynb)). I used these subclasses when doing the CLIP zero-shot classification, then aggregated the subclass scores back into the desired classes.

    This lead me to take a two step approach, where first I find all items that appear in the image with CLIP, then I pick the images where the top CLIP predicted category had a probability of over 80%. I then grouped by the predicted category and concatenated the corresponding descriptions into a text block for each class. Then, I can take a description for an ambiguous item (not in the 80%+ set), embed it, and compare the embedding to these labeled descriptions. The rationale for this is that, in and of themselves, the text descriptions weren't directly informative to the underlying class. Instead, they had words that correlated with the underlying classes (hardware -> bag, sleeves -> tops, etc.). By creating these description mappings (class -> block of descriptions), we can hopefully pickup on these correlations. After comparing the description to the labeled block, I get a text-only softmax distribution over the classes.


    To get my finalized prediction, I took my CLIP probabilities and multiplied them by my miniLM class probabilities and renormalized. (Interestingly this only changed ~60 of the class predictions from just the CLIP predictions. Given more time, I could do a better ensembling than just a multiplication of the probabilities to produce better results.)
    

2. What are the aspects that you considered when designing?
    
    The first things I consider when designing any ML system is how I would approach the problem by hand. When looking through some examples, I found myself having difficulty telling what the correct class is given the photo or the description alone. What I would do by hand is look at the image, see all of the items that were in it (if there was more than one), then use the text to disambiguate. This process inspired how the model was designed. 

    I also had to consider how the classes were balanced (especially in my CLIP 80%+ labeled set). For example, there were no jumpsuit predictions in that set, which meant that no descriptions were collected. This meant that miniLM would never compare a new description to one of a known jumpsuit and therefore would assign no weight to it.

    I had to worry about overfitting to the data, given the small number of examples. Because of this I created my subclasses not from looking at the actual data, but from external sources.


3. What are the cases your solution covers, how are they covered and why are they important?
    
    My solution covers cases in which either the image consists of a single item (if that item is in one of my enumerated subclasses), or if it does not, ones in which the text can disambiguate between the items in the image. If the image contains just one item, CLIP should correctly label it. If it contains more than one, and the description is more similar to the labeled descriptions of the correct class, miniLM should help to disambiguate.

    These cases are important firstly because they seem to be the majority of the input set. Furthermore, these cases are scalable, by adding new subclasses which will improve the accuracy of the CLIP predictions, and therefore the bootstrapped miniLM predicitions.

4. What are the cases your solution does not cover and what are the ways you can extend your current solution for them?

    My solution does not cover instances where CLIP is incorrectly confident about the main item in an image. The text seems to be not informative enough to move the predictions that much. This solution could definitely be extended with a better ensembling method between the image predictions and the text predictions. This might look like just a simple weighting, or a whole model that takes the image_predicitons and text_predictions and outputs the ensembled_predictions, though this would also require a labeled set to train.

    My solution does not cover cases in which the image does not contain one of the subclasses I created. These subclasses have to be comprehensive in order to get good CLIP predictions. These can be iteratively updated based on hand labeled data.
    
    My solution does not cover various edge cases, like when the description is empty, or the photo doesnt actually contain the desired class prediction. This could be solved by not doing this CLIP based bootstrap approach, but instead make a hand labeled golden set. Then from this set, we could train a model independently on text and image to get a more robust prediction for these cases.
    
