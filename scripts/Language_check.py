from langdetect import detect
import pandas as pd
from deep_translator import GoogleTranslator

# Reading Data
#df_analysis = pd.read_csv("youtube_shorts_analysis.csv")

# Language detection (if error occurs, defaults to 'unknown')
def detect_language(text):
    try:
        return detect(text)
    except:
        return "unknown"

# Run language detection
df_analysis["language"] = df_analysis["title"].astype(str).apply(detect_language)

# Statistical language distribution
print(df_analysis["language"].value_counts())

# Save data
#df_analysis.to_csv("youtube_shorts_with_language.csv", index=False)
#print("Language detection completed!")


# Read data
#df_analysis = pd.read_csv("youtube_shorts_with_language.csv")

# Translate only non-English titles
def translate_text(text, lang):
    if lang == "en":
        return text  
# If it is in English, do not translate
    try:
        return GoogleTranslator(source=lang, target="en").translate(text)
    except:
        return text  # If translation fails, keep the original text

# Add Translation
df_analysis["title_translated"] = df_analysis.apply(lambda row: translate_text(row["title"], row["language"]), axis=1)

# Save Data if you want
#df_analysis.to_csv("youtube_shorts_translated.csv", index=False)
#print("Translation completed, all titles have been converted to English!")
