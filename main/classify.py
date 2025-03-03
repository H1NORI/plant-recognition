from fastai.vision.all import *
from fastai.vision.utils import *
import matplotlib.pyplot as plt
import cv2



learn2 = load_learner('plantsModel.pkl')
categories = (
    "African Violet (Saintpaulia ionantha)",
    "Aloe Vera",
    "Anthurium (Anthurium andraeanum)",
    "Areca Palm (Dypsis lutescens)",
    "Asparagus Fern (Asparagus setaceus)",
    "Begonia (Begonia spp.)",
    "Bird of Paradise (Strelitzia reginae)",
    "Birds Nest Fern (Asplenium nidus)",
    "Boston Fern (Nephrolepis exaltata)",
    "Calathea",
    "Cast Iron Plant (Aspidistra elatior)",
    "Chinese evergreen (Aglaonema)",
    "Chinese Money Plant (Pilea peperomioides)",
    "Christmas Cactus (Schlumbergera bridgesii)",
    "Chrysanthemum",
    "Ctenanthe",
    "Daffodils (Narcissus spp.)",
    "Dracaena",
    "Dumb Cane (Dieffenbachia spp.)",
    "Elephant Ear (Alocasia spp.)",
    "English Ivy (Hedera helix)",
    "Hyacinth (Hyacinthus orientalis)",
    "Iron Cross begonia (Begonia masoniana)",
    "Jade plant (Crassula ovata)",
    "Kalanchoe",
    "Lilium (Hemerocallis)",
    "Lily of the valley (Convallaria majalis)",
    "Money Tree (Pachira aquatica)",
    "Monstera Deliciosa (Monstera deliciosa)",
    "Orchid",
    "Parlor Palm (Chamaedorea elegans)",
    "Peace lily",
    "Poinsettia (Euphorbia pulcherrima)",
    "Polka Dot Plant (Hypoestes phyllostachya)",
    "Ponytail Palm (Beaucarnea recurvata)",
    "Pothos (Ivy arum)",
    "Prayer Plant (Maranta leuconeura)",
    "Rattlesnake Plant (Calathea lancifolia)",
    "Rubber Plant (Ficus elastica)",
    "Sago Palm (Cycas revoluta)",
    "Schefflera",
    "Snake plant (Sanseviera)",
    "Tradescantia",
    "Tulip",
    "Venus Flytrap",
    "Yucca",
    "ZZ Plant (Zamioculcas zamiifolia)"
)


def classify_image(img_path, threshold=0.7):
    learn2.model.eval()

    imgToPredict = PILImage.create(img_path)
    pred = learn2.predict(imgToPredict)
    pred_probs_dec = {cat: f"{p:.20f}" for cat, p in zip(learn2.dls.vocab, pred[2].tolist())}
    max_prob = max(pred[2].tolist())

    if max_prob >= threshold:
        return pred[0], pred_probs_dec, max_prob
    else:
        return "Uncertain", pred_probs_dec, max_prob


def getClassesForLearner(learner):
    return learner.dls.vocab


def getLearner():
    return learn2


def getCategories():
    return categories

def getProbsInDecimal(pred):
    pred_probs_dec = [f"{p:.20f}" for p in pred[2].tolist()]
    return pred_probs_dec