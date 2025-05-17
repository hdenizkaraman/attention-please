import os
# Model Info
MODEL:str = "dbmdz/bert-base-turkish-cased"
DATASET:str = "denizzhansahin/Turkish_News-2024"
DATASET_FIELD:str = "Baslik"

# Constants
STOPWORDS: set[str] = {
    "ve", "bir", "bu", "da", "de", "için", "ile", "mi", "ne", "ama", "veya",
    "gibi", "çok", "az", "daha", "en", "ki", "o", "şu", "biz", "siz", "onlar",
    "ben", "sen", "mı", "mu", "mü", "müsün", "miyim", "ise", "diye", "ya", "çünkü",
    "hem", "her", "hiç", "kadar", "sanki", "şey", "şöyle", "tüm", "yani"
}