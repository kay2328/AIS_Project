"""
To perform initial exploratory analysis on the VizWiz image captioning dataset.
It loads the dataset, computes statistics (e.g. caption lengths),
visualizes the image and captions distributions to 
understand the dataset. The graph outputs can be found under image_description/outputs/ folder.
"""
## Import necessay libraries
import os
import json
import re
import logging
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import yaml
import os
from collections import Counter
import nltk
import cv2
from wordcloud import WordCloud
import warnings
warnings.filterwarnings(action='ignore')



def load_config(config_path="config_path.yaml"):
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config

# Load configuration and expose variables
config = load_config()
TRAIN_JSON = config["data"]["train_json"]
VALID_JSON = config["data"]["valid_json"]
TEST_JSON = config["data"]["test_json"]
EXPLORATION_OUTPUT = config["output"]["exploration"]
DATASET_PATH = config["data"]["img_dataset"]


## Set up for logging
def setup_logging():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")



## Load the data from the annotations files where the dataset details are available in .json format
def load_data(data_path: str) -> list:
    if not os.path.exists(data_path):
        logging.error(f"Data file {data_path} not found!")
        raise FileNotFoundError(f"Data file {data_path} not found!")
    with open(data_path, "r") as f:
        data = json.load(f)
    logging.info(f"Loaded {list(data.keys())} details from {data_path}")
    return data



def prepare_data(data: dict, data_path: str, type: str):
    df_path = os.path.join(data_path, f"{type}.csv")
    image_df = pd.DataFrame(data["images"])
    if type != "test":   
        annot_df = pd.DataFrame(data["annotations"]) 
        if annot_df.empty or image_df.empty:
            logging.error("One of the datasets is empty. Check your JSON files.")
            raise ValueError("Images or annotations is empty.")
        else:
            logging.info(f"Total image samples in {type} data: {len(image_df)}")
            logging.info(f"Total annotations/captions samples in {type} data: {len(annot_df)}")
        df = pd.merge(annot_df, image_df, left_on="image_id", right_on="id", suffixes=('_annot', '_img'))
        logging.info(f"Merged Images and Annotations DataFrame shape: {df.shape}")
    else:
        df = image_df
    df.to_csv(df_path, index=False)
    logging.info(f"Saving the DataFrame as csv in : {df_path}")
    return df



def img_annot_count(df, output_dir, type):
    logging.info("Starting Image-Annotation Count Analysis...")
    if "image_id" in df.columns:
        images_count = df['image_id'].nunique()  # unique images
        captions_count = df.shape[0]         # total annotations (captions)
        logging.info(f"Unique images in {type} set: {df['image_id'].nunique()}")
        # Create a count DataFrame for plotting
        count_df = pd.DataFrame({'Category': ['Images', 'Captions'], 'Count': [images_count, captions_count]})
        
        # Plot using seaborn's barplot
        sns.set_theme(style="whitegrid")
        plt.figure(figsize=(8, 6))
        ax = sns.barplot(x="Category", y="Count", data=count_df, palette="viridis")
        ax.set_title(f"Images and Annotations Distribution - {type} set")
        ax.set_xlabel("Category")
        ax.set_ylabel("Count")
        # Annotate each bar with its count value
        for p in ax.patches:
            height = p.get_height()
            ax.annotate(f'{int(height)}', (p.get_x() + p.get_width() / 2., height), ha='center', va='bottom')
        plt.tight_layout()
        # Saving the plot
        img_annot_plot_path = os.path.join(output_dir, f"{type}_img_annot_distribution.png")
        plt.savefig(img_annot_plot_path)
        logging.info(f"Images and Annotations distribution plot saved to {img_annot_plot_path}")
        plt.close()
    return count_df



def image_analysis(df, image_folder, output_dir, type):
    logging.info("Starting Image Analysis...")
    resolutions = []
    blurriness_scores = []
    missing_images = []
    files_lst = df["file_name"].unique()
    logging.info(f"Number of image files: {len(files_lst)}")
    for i in range(len(files_lst)):
        image_file = files_lst[i]
        image_path = os.path.join(image_folder, image_file)
        img = cv2.imread(image_path)
        if image_file and os.path.exists(image_path) and not img is None:
            h, w = img.shape[:2]
            resolutions.append((w, h))
            # Blurriness: variance of the Laplacian (lower value -> blurrier image)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            variance = cv2.Laplacian(gray, cv2.CV_64F).var()
            blurriness_scores.append(variance)
        else:
            missing_images.append(image_path)

    # Plot resolution distributions
    if resolutions:
        widths = [w for w, h in resolutions]
        heights = [h for w, h in resolutions]
        plt.figure(figsize=(8, 6))
        # Plot the widths with one color and some transparency
        sns.histplot(widths, kde=True, color='blue', label='Images Width', bins=30, alpha=0.5)
        # Plot the heights with another color and transparency
        sns.histplot(heights, kde=True, color='red', label='Images Height', bins=30, alpha=0.5)
        plt.legend(title="Dimension")
        plt.title(f"Image Dimension Distribution - {type} set")
        plt.xlabel("Pixel Value")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{type}_image_dimensions_distribution.png"))
        plt.close()
        
    # Plot blurriness distribution
    if blurriness_scores:
        plt.figure(figsize=(8,6))
        sns.histplot(blurriness_scores, kde=True)
        plt.title("Image Blurriness Distribution")
        plt.xlabel("Blurriness Measure")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{type}_image_blurriness_distribution.png"))
        plt.close()
    logging.info(f"Processed {len(resolutions)} images; missing/corrupt images: {len(missing_images)}")
    print(df.columns)
    # plot text_detected distribution
    col = "text_detected" if type == "test" else "text_detected_img"
    flag_counts = df[col].value_counts()
    logging.info(f"text_detected distribution in {type} set:")
    logging.info(flag_counts.to_string())
    plt.figure(figsize=(8, 6))
    ax = sns.barplot(x=flag_counts.index.astype(str), y = flag_counts.values, palette="viridis")
    ax.set_title(f"text_detected Distribution - {type} set")
    ax.set_xlabel("text_detected")
    ax.set_ylabel("Count")
    # Annotate each bar with its count value
    for p in ax.patches:
        height = p.get_height()
        ax.annotate(f'{int(height)}', (p.get_x() + p.get_width() / 2., height), ha='center', va='bottom')        
    plt.tight_layout()
    flag_plot_path = os.path.join(output_dir, f"{type}_text_detected_distribution.png")
    plt.savefig(flag_plot_path)
    logging.info(f"text_detected distribution plot saved to {flag_plot_path}")
    plt.close()
    return missing_images, resolutions, blurriness_scores




def is_caption_corrupt(caption):
    caption = caption.lower()
    # 1. Check for None or non-string
    if not caption:
        return True
    # Strip whitespace
    c = caption.strip()
    # 2. Check placeholder patterns
    # Looking for any string that starts with '#' or equals '???' or something suspicious/invalid
    if re.match(r"^#.*", c) or c.lower() in ["???", "na", "null"]:
        return True
    # 3. Check if it contains at least one alphabetic character
    # This helps to exclude purely numeric or symbol-only captions
    if not re.search(r"[a-zA-Z]", c):
        return True
    return False


def caption_analysis(df, output_dir, type):
    # 1. Caption corrupt analysis
    df["is_corrupt"] = df["caption"].apply(is_caption_corrupt)
    logging.info(f"Number of corrupted captions: {df[df["is_corrupt"]==True].shape[0]}")
    # Removing the corrupted captions from dataset if present
    df = df[df["is_corrupt"]==False].reset_index(drop=True)
    logging.info(f" Number of valid data: {df.shape[0]}")
    # 2. Caption length analysis
    logging.info("Starting Caption Length Analysis...")
    df['caption_length'] = df['caption'].apply(lambda x: len(x.split()))
    logging.info("Caption length statistics:")
    logging.info(df['caption_length'].describe().to_string())
    # Histogram plot of the captions length distribution
    plt.figure(figsize=(8, 6))
    plt.hist(df['caption_length'], bins=range(1, df['caption_length'].max() + 2), edgecolor="black")
    plt.title(f"Caption Length Distribution - {type} set", size=15)
    plt.xlabel("Number of Words")
    plt.ylabel("Frequency")
    plt.tight_layout()
    # Saving the plot
    caption_plot_path = os.path.join(output_dir, f"{type}_caption_length_distribution.png")
    plt.savefig(caption_plot_path)
    logging.info(f"Caption length distribution plot saved to {caption_plot_path}")
    plt.close()  

    # 3. Vocabulary Analysis of the captions
    logging.info("Starting Caption Vocabulary Analysis...")
    all_captions = " ".join(df['caption'].tolist()).lower()
    
    # Word cloud visualization
    wordcloud = WordCloud(width=800, height=600, background_color="white", colormap="viridis", max_words=200).generate(all_captions)
    # Plot the word cloud
    plt.figure(figsize=(10, 8))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.title(f"Word Cloud - {type} set", size=15)
    plt.tight_layout()
    # Save the word cloud image
    vocab_plot_path = os.path.join(output_dir, f"{type}_vocabulary_wordcloud.png")
    plt.savefig(vocab_plot_path)
    logging.info(f"Vocabulary word cloud saved to {vocab_plot_path}")
    plt.close() 
    
    tokens = nltk.word_tokenize(all_captions)
    word_freq = Counter(tokens)
    top_words = word_freq.most_common(30)
    vocab_df = pd.DataFrame(top_words, columns=["Word", "Frequency"])
    plt.figure(figsize=(8, 6))
    ax = sns.barplot(x="Frequency", y="Word", data=vocab_df, palette="viridis")
    ax.set_title(f"Top 30 Most Frequent Words - {type} set")
    plt.tight_layout()
    vocab_plot_path = os.path.join(output_dir, f"{type}_vocabulary_distribution.png")
    plt.savefig(vocab_plot_path)
    logging.info(f"Vocabulary distribution plot saved to {vocab_plot_path}")
    plt.close() 



def quality_flags_analysis(df, output_dir, type):
    print(df.columns)
    logging.info("Starting Quality Flag Analysis...")
    quality_flags = [col for col in ['is_rejected', 'is_precanned'] if col in df.columns]
    if quality_flags:
        for flag in quality_flags:
            flag_counts = df[flag].value_counts()
            logging.info(f"{flag} distribution in {type} set:")
            logging.info(flag_counts.to_string())
            plt.figure(figsize=(8, 6))
            ax = sns.barplot(x=flag_counts.index.astype(str), y = flag_counts.values, palette="viridis")
            ax.set_title(f"{flag} Distribution - {type} set")
            ax.set_xlabel(flag)
            ax.set_ylabel("Count")
            # Annotate each bar with its count value
            for p in ax.patches:
                height = p.get_height()
                ax.annotate(f'{int(height)}', (p.get_x() + p.get_width() / 2., height), ha='center', va='bottom')        
            plt.tight_layout()
            flag_plot_path = os.path.join(output_dir, f"{type}_{flag}_distribution.png")
            plt.savefig(flag_plot_path)
            logging.info(f"{flag} distribution plot saved to {flag_plot_path}")
            plt.close()
    return flag_counts



def explore_data(df, image_folder, output_dir, type):
    if type == "test":
        # Only image analysis
        mis_imgs, res, blur = image_analysis(df, image_folder, output_dir, type)
    else:
        # 1. Image - Annotation Count analysis
        img_annot_df =  img_annot_count(df, output_dir, type)
        # 2. Images analysis
        mis_imgs, res, blur = image_analysis(df, image_folder, output_dir, type)
        # 3. Caption length analysis
        caption_analysis(df, output_dir, type)
        # 4. Data Quality Flags Analysis   
        flag_df = quality_flags_analysis(df, output_dir, type) 
    


def main():
    setup_logging()
    os.makedirs(EXPLORATION_OUTPUT, exist_ok=True)
    for type, json_file in zip(["train" ,"valid", "test"], [TRAIN_JSON, VALID_JSON, TEST_JSON]):
    #for type, json_file in zip(["train"], [TRAIN_JSON]):
        image_path = os.path.join(DATASET_PATH, type)
        data = load_data(json_file)
        image_annot_data = prepare_data(data, DATASET_PATH, type)
        explore_data(image_annot_data, image_path, EXPLORATION_OUTPUT, type)

if __name__ == "__main__":
    main()

"""
The annotations file consists of three main details: info, images and annotations in .json format for train, test and validation sets
Since the image file are large in size that cannot be loaded directly, the annotations file with all the details about the dataset with images and captions 
is loaded as a dataframe and used for exploration

"""
