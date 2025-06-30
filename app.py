from flask import Flask, render_template, url_for, request, flash
import tweepy
import re, string, csv, pickle, os
from os.path import join, dirname, realpath
import pandas as pd
import numpy as np
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory, StopWordRemover, ArrayDictionary
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from googletrans import Translator
from textblob import TextBlob
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from PIL import Image
import urllib.request
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn.naive_bayes import MultinomialNB
nltk.download('punkt')
nltk.download('stopwords')


#Preprocessing Twitter
hasil_preprocessing  = []

def preprocessing_twitter():
    from googletrans import Translator
    translator = Translator()
    hasil_preprocessing.clear()

    with open('static/files/Data Preprocessing.csv', 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow([
            'Tanggal', 'Username', 'Tweet', 
            'Cleansing', 'Case Folding', 'Normalisasi',
            'Tokenizing', 'Stopword', 'Stemming', 'Translate'
        ])

        processed_casefolds = set()  # untuk menghindari duplikat

        with open("static/files/Data Scraping.csv", "r", encoding='utf-8') as csvfile:
            readCSV = csv.reader(csvfile, delimiter=',')

            for row in readCSV:
                tanggal = row[0]
                username = row[1]
                tweet = row[2]

                # --- CLEANSING ---
                clean = tweet
                clean = re.sub(r'@[A-Za-z0-9_]+', '', clean)
                clean = re.sub(r'#\w+', '', clean)
                clean = re.sub(r'RT[\s]+', '', clean)
                clean = re.sub(r'https?://\S+', '', clean)
                clean = re.sub(r'[^A-Za-z0-9 ]', '', clean)
                clean = re.sub(r'\s+', ' ', clean).strip()

                # --- CASE FOLDING ---
                casefold = clean.lower()

                # === SKIP JIKA DUPLIKAT ===
                if casefold in processed_casefolds:
                    continue
                processed_casefolds.add(casefold)

                # --- NORMALISASI ---
                normalized = normalize_text(casefold)

                # --- TOKENIZING ---
                tokenizing = nltk.tokenize.word_tokenize(normalized)

                # --- STOPWORD REMOVAL ---
                stop_factory = StopWordRemoverFactory().get_stop_words()
                more_stop_word = ['tidak']
                all_stopwords = stop_factory + more_stop_word
                dictionary = ArrayDictionary(all_stopwords)
                stopword_remover = StopWordRemover(dictionary)
                stop_removed_text = stopword_remover.remove(normalized)
                stopword_tokens = nltk.tokenize.word_tokenize(stop_removed_text)

                # --- STEMMING ---
                stemming = stemming_tokens(stopword_tokens)

                # --- TRANSLATE ---
                try:
                    translation = translator.translate(stemming, dest='en')
                    translated = translation.text.lower()
                except:
                    translated = "terjemahan gagal"

                # --- SIMPAN SEMUA HASIL ---
                tweets = [
                    tanggal, username, tweet,
                    clean, casefold, normalized,
                    ' '.join(tokenizing),
                    ' '.join(stopword_tokens),
                    stemming,
                    translated
                ]
                hasil_preprocessing.append(tweets)
                writer.writerow(tweets)

    flash('Preprocessing Berhasil', 'preprocessing_data')

def normalize_text(text):
    substitutions = {
        'sdh': 'sudah', ' yg ': ' yang ', ' nggak ': ' tidak ', ' gak ': ' tidak ',
        ' bangetdari ': ' banget dari ', 'vibes ': 'suasana ', 'mantab ': 'mantap ',
        ' benarsetuju ': ' benar setuju ', ' ganjarmahfud ': ' ganjar mahfud ',
        ' stylish ': ' bergaya ', ' ngapusi ': ' bohong ', ' gede ': ' besar ',
        ' all in ': ' yakin ', ' blokkkkk ': ' goblok ', ' blokkkk ': ' goblok ',
        ' blokkk ': ' goblok ', ' blokk ': ' goblok ', ' blok ': ' goblok ',
        ' ri ': ' republik indonesia ', ' kem3nangan ': ' kemenangan ',
        ' sat set ': ' cepat ', ' ala ': ' dari ', ' best ': ' terbaik ',
        ' bgttt ': ' banget ', ' gue ': ' saya ', ' hrs ': ' harus ',
        ' fixed ': ' tetap ', ' blom ': ' belum ', ' aing ': ' aku ',
        ' tehnologi ': ' teknologi ', ' jd ': ' jadi ', ' dg ': ' dengan ',
        ' kudu ': ' harus ', ' jk ': ' jika ', ' problem ': ' masalah ',
        ' iru ': ' itu ', ' duit ': ' uang ', ' duid ': ' uang ',
        ' bgsd ': ' bangsat ', ' jt ': ' juta ', ' stop ': ' berhenti ',
        ' ngeri ': ' seram ', ' turu ': ' tidur ', ' early ': ' awal ',
        ' pertamna ': ' pertamina ', ' mnurut ': ' menurut ', ' trus ': ' terus ',
        ' msh ': ' masih ', ' simple ': ' mudah ', ' worth ': ' layak ',
        ' hny ': ' hanya ', ' dn ': ' dan ', ' jln ': ' jalan ', ' bgt ': ' banget ',
        ' ga ': ' tidak ', ' text ': ' teks ', ' end ': ' selesai ',
        ' kelen ': ' kalian ', ' tuk ': ' untuk ', ' kk ': ' kakak '
    }
    for key, val in substitutions.items():
        text = re.sub(key, val, text)
    return text

def stemming_tokens(tokens):
    stemmer = StemmerFactory().create_stemmer()
    return ' '.join([stemmer.stem(token) for token in tokens])


# Labeling 5 Kelas
hasil_labeling = []

def labeling_twitter():
    hasil_labeling.clear()

    with open("static/files/Data Preprocessing.csv", "r", encoding='utf-8') as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        next(readCSV)  # Lewati header CSV

        with open('static/files/Data Labeling.csv', 'w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)

            # Header file hasil labeling
            writer.writerow(['Tanggal', 'Username', 'Tweet', 'Stemming', 'Translate', 'Label'])

            for row in readCSV:
                tanggal = row[0]
                username = row[1]
                tweet_asli = row[2]
                stemming = row[8]
                translated = row[9]  # hasil translate

                # Lewati jika terjemahan gagal atau kosong
                if translated.lower() == "terjemahan gagal" or translated.strip() == "":
                    continue

                try:
                    analysis = TextBlob(translated)
                    score = analysis.sentiment.polarity
                except Exception as e:
                    continue  # Lewati jika error analisis sentimen

                # Penentuan label berdasarkan polaritas
                if score >= 0.6:
                    label = 'Sangat Mendukung'
                elif 0.2 <= score < 0.6:
                    label = 'Mendukung'
                elif -0.2 < score < 0.2:
                    label = 'Netral'
                elif -0.6 < score <= -0.2:
                    label = 'Tidak Mendukung'
                else:
                    label = 'Sangat Tidak Mendukung'

                hasil = [tanggal, username, tweet_asli, stemming, translated, label]
                hasil_labeling.append(hasil)
                writer.writerow(hasil)

    flash('Labeling 5 Kelas Berhasil', 'labeling_data')

#Klasifikasi

# Membuat variabel df
df = None
df2 = None

# menentukan akurasi 0
akurasi = 0

def proses_klasifikasi():
    global df, df2, akurasi
    tweet = []
    y = []

    # Baca data labeling
    with open("static/files/Data Labeling.csv", encoding='utf-8') as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        next(readCSV)  # Lewati header

        for row in readCSV:
            tweet_text = row[4]  # kolom Translate
            label = row[5]       # kolom Label

            if tweet_text.lower() != "terjemahan gagal" and tweet_text.strip() != "":
                tweet.append(tweet_text)
                y.append(label)

    # TF-IDF
    vectorizer = TfidfVectorizer()
    x = vectorizer.fit_transform(tweet)

    # Split data training dan testing
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42
    )

    # Naive Bayes
    clf = MultinomialNB()
    clf.fit(x_train, y_train)
    predict = clf.predict(x_test)

    # Simpan classification report ke CSV
    report = classification_report(y_test, predict, output_dict=True)
    clsf_report = pd.DataFrame(report).transpose()
    clsf_report.to_csv('static/files/Data Klasifikasi.csv', index=True)

    # Simpan model dan vectorizer
    pickle.dump(vectorizer, open('static/files/vec.pkl', 'wb'))
    pickle.dump(x, open('static/files/tfidf.pkl', 'wb'))
    pickle.dump(clf, open('static/files/model.pkl', 'wb'))

    # Confusion Matrix
    unique_label = sorted(list(set(y)))  # gunakan semua label unik dari data
    cmtx = pd.DataFrame(
        confusion_matrix(y_test, predict, labels=unique_label),
        index=[f'pred:{x}' for x in unique_label],
        columns=[f'true:{x}' for x in unique_label]
    )
    cmtx.to_csv('static/files/Data Confusion Matrix.csv', index=True)

    # Baca ulang hasil evaluasi
    df = pd.read_csv('static/files/Data Confusion Matrix.csv', sep=",")
    df.rename(columns={'Unnamed: 0': ''}, inplace=True)

    df2 = pd.read_csv('static/files/Data Klasifikasi.csv', sep=",")
    df2.rename(columns={'Unnamed: 0': ''}, inplace=True)

    # Hitung akurasi
    akurasi = round(accuracy_score(y_test, predict) * 100, 2)

    # Wordcloud
    kalimat = " ".join(tweet)
    urllib.request.urlretrieve(
        "https://firebasestorage.googleapis.com/v0/b/sentimen-97d49.appspot.com/o/Circle-icon.png?alt=media&token=b9647ca7-dfdb-46cd-80a9-cfcaa45a1ee4",
        'circle.png')
    mask = np.array(Image.open("circle.png"))
    wordcloud = WordCloud(width=1600, height=800,
                          max_font_size=200, background_color='white', mask=mask)
    wordcloud.generate(kalimat)
    plt.figure(figsize=(12, 10))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.savefig('static/files/wordcloud.png')


    # Diagram Batang Distribusi Sentimen
    label_series = pd.Series(y)
    plt.figure(figsize=(8, 5))
    label_series.value_counts().plot(kind='bar', color='skyblue')
    plt.title('Distribusi Sentimen')
    plt.xlabel('Label')
    plt.ylabel('Jumlah')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('static/files/bar_sentimen.png')
    plt.close()

    plt.figure(figsize=(6, 6))
    colors = plt.cm.Pastel1.colors
    counts = label_series.value_counts()
    labels = counts.index
    sizes = counts.values

    # Hitung persentase manual
    total = sum(sizes)
    label_with_pct = [f"{label} — {round((size/total)*100, 1)}%" for label, size in zip(labels, sizes)]

    # Pie chart tanpa label dan tanpa persentase
    wedges, texts = plt.pie(
    sizes,
    startangle=90,
    colors=colors,
        )

    # Tambahkan legend di samping
    plt.legend(wedges, label_with_pct, title="Label Sentimen", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))

    # Opsional: Donut chart style
    centre_circle = plt.Circle((0, 0), 0.70, fc='white')
    fig = plt.gcf()
    fig.gca().add_artist(centre_circle)

    plt.title('Proporsi Sentimen')
    plt.tight_layout()
    plt.savefig('static/files/pie_sentimen.png')
    plt.close()

    flash('Klasifikasi Berhasil', 'klasifikasi_data')



app = Flask(__name__)
app.config['SECRET_KEY'] = 'farez'


# Upload folder
UPLOAD_FOLDER = 'static/files'
ALLOWED_EXTENSION = set(['csv'])
app.config['UPLOAD_FOLDER'] =  UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSION

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/preprocessing', methods=['GET', 'POST'])
def preprocessing():
    if request.method == 'POST':
        if request.form.get('upload') == 'Upload Data':
            hasil_preprocessing.clear()
            file = request.files['file']
            if not allowed_file(file.filename):
                flash('Format file tidak diperbolehkan', 'upload_gagal')
                return render_template('preprocessing.html', value=hasil_preprocessing)

            if 'file' not in request.files:
                flash('File tidak boleh kosong', 'upload_gagal')
                return render_template('preprocessing.html', value=hasil_preprocessing)

            if file.filename == '':
                flash('File tidak boleh kosong', 'upload_gagal')
                return render_template('preprocessing.html', value=hasil_preprocessing)

            if file and allowed_file(file.filename):
                file.filename = "Data Scraping.csv"
                file.save(os.path.join(app.config['UPLOAD_FOLDER'], file.filename))
                hasil_preprocessing.clear()
                flash('File Berhasil di upload', 'upload_berhasil')
                return render_template('preprocessing.html')

        if request.form.get('preprocess') == 'Preprocessing Data':
            preprocessing_twitter()
            return render_template('preprocessing.html', value=hasil_preprocessing)

    return render_template('preprocessing.html', value=hasil_preprocessing)

@app.route('/labeling', methods=['GET', 'POST'])
def labeling():
    if request.method == 'POST':
        if request.form.get('upload') == 'Upload Data':
            hasil_labeling.clear()
            file = request.files['file']
            if not allowed_file(file.filename):
                flash('Format file tidak diperbolehkan', 'upload_gagal')
                return render_template('labeling.html', value=hasil_labeling)

            if 'file' not in request.files:
                flash('File tidak boleh kosong', 'upload_gagal')
                return render_template('labeling.html', value=hasil_labeling)

            if file.filename == '':
                flash('File tidak boleh kosong', 'upload_gagal')
                return render_template('labeling.html', value=hasil_labeling)

            if file and allowed_file(file.filename):
                file.filename = "Data Preprocessing.csv"
                file.save(os.path.join(app.config['UPLOAD_FOLDER'], file.filename))
                hasil_labeling.clear()
                flash('File Berhasil di upload', 'upload_berhasil')
                return render_template('labeling.html')

        if request.form.get('labeling') == 'Labeling Data':
            labeling_twitter()
            return render_template('labeling.html', value=hasil_labeling)
            
    return render_template('labeling.html', value=hasil_labeling)

@app.route('/klasifikasi', methods=['GET', 'POST'])
def klasifikasi():
    if request.method == 'POST':
        if request.form.get('upload') == 'Upload Data':
            file = request.files['file']
            if not allowed_file(file.filename):
                flash('Format file tidak diperbolehkan', 'upload_gagal')
                return render_template('klasifikasi.html')
            if 'file' not in request.files:
                flash('File tidak boleh kosong', 'upload_gagal')
                return render_template('klasifikasi.html',)
            if file.filename == '':
                flash('File tidak boleh kosong', 'upload_gagal')
                return render_template('klasifikasi.html')
            if file and allowed_file(file.filename):
                file.filename = "Data Labeling.csv"
                file.save(os.path.join(app.config['UPLOAD_FOLDER'], file.filename))
                flash('File Berhasil di upload', 'upload_berhasil')
                return render_template('klasifikasi.html')

        if request.form.get('klasifikasi') == 'Klasifikasi Data':
            proses_klasifikasi()
            return render_template('klasifikasi.html', accuracy=akurasi, tables=[df.to_html(classes='table table-bordered', index=False, justify='left')], titles=df.columns.values, tables2=[df2.to_html(classes='table table-bordered', index=False, justify='left')], titles2=df2.columns.values)
            
    if akurasi == 0:
        return render_template('klasifikasi.html')
    else:
        return render_template('klasifikasi.html', accuracy=akurasi, tables=[df.to_html(classes='table table-bordered', index=False, justify='left')], titles=df.columns.values, tables2=[df2.to_html(classes='table table-bordered', index=False, justify='left')], titles2=df2.columns.values)

@app.route('/visualisasi')
def visualisasi():
    return render_template('visualisasi.html')

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port, debug=False)