# Core Packages
import string
from collections import Counter
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import tkinter as tk
from tkinter import *
from tkinter import ttk
from tkinter.scrolledtext import *
import tkinter.filedialog
from transformers import pipeline
import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import numpy as np
import seaborn  as sns
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt1
from wordcloud import WordCloud
from wordcloud import STOPWORDS

emotion = pipeline('sentiment-analysis', model='arpanghoshal/EmoRoBERTa')


 
 # Structure and Layout
window = Tk()
window.title("Sentiment and Emotion analysis")
window.geometry("1000x400")
window.config(background='black')

# TAB LAYOUT
tab_control = ttk.Notebook(window)
 
tab1 = ttk.Frame(tab_control)
tab2 = ttk.Frame(tab_control)
tab3 = ttk.Frame(tab_control)

# ADD TABS TO NOTEBOOK
tab_control.add(tab1, text='Text sentiment and emotion analysis')
tab_control.add(tab2, text='File sentiment and emotion analysis ')
tab_control.add(tab3, text='About')


label1 = Label(tab1, text= 'NLP Made Simple',padx=5, pady=5)
label1.grid(column=0, row=0)
 
label2 = Label(tab2, text= 'File Processing',padx=5, pady=5)
label2.grid(column=0, row=0)

label3 = Label(tab3, text= 'About',padx=5, pady=5)
label3.grid(column=0, row=0)

tab_control.pack(expand=1, fill='both')

about_label = Label(tab3,text="emotions and sentiment analysis based on text and file \n  ",pady=5,padx=5)
about_label.grid(column=0,row=1)

# Functions FOR NLP  FOR TAB ONE




def get_sentiment():
	live_text = str(raw_entry.get())
	lower_case = live_text.lower()
	cleaned_text = lower_case.translate(str.maketrans('', '', string.punctuation))
	# Using word_tokenize because it's faster than split()
	tokenized_words = word_tokenize(cleaned_text, "english")

	# Removing Stop Words
	final_words = []
	for word in tokenized_words:
		if word not in stopwords.words('english'):
			final_words.append(word)
	print(final_words)

	# Lemmatization - From plural to single + Base form of a word (example better-> good)
	lemma_words = []
	for word in final_words:
		word = WordNetLemmatizer().lemmatize(word)
		lemma_words.append(word)
	print(lemma_words)
	cleaned_text=" ".join(lemma_words)

	def sentiment_analyse(sentiment_text):
		score = SentimentIntensityAnalyzer().polarity_scores(sentiment_text)
		list = [(k, v) for k, v in score.items()]
		for a, b in list:
			result = '\n  {} : {}  \n'.format(a,b)
			tab1_display.insert(tk.END, result)


		if score['compound'] <= - 0.05:
			tab1_display.insert(tk.END, "\nNegative Sentiment\n")
		elif score['compound'] >= 0.05:
			tab1_display.insert(tk.END, "\nPositive Sentiment\n")
		else:
			tab1_display.insert(tk.END, "\nNeutral Sentiment\n")


	sentiment_analyse(cleaned_text)

def get_emotion():
	live_emotion = str(raw_entry.get())
	lower_case = live_emotion.lower()
	cleaned_text = lower_case.translate(str.maketrans('', '', string.punctuation))
	emotion_labels = emotion(cleaned_text)
	result=emotion_labels[0]['label']

	tab1_display.insert(tk.END, result)


def word_cloud():
	live_text = str(raw_entry.get())
	cleaned_text = live_text.translate(str.maketrans('', '', string.punctuation))
	word_cloud = WordCloud(collocations=False, background_color='white').generate(cleaned_text)
	plt.imshow(word_cloud, interpolation='bilinear')
	plt.axis("off")
	plt.show()


# Clear entry widget
def clear_entry_text():
	entry1.delete(0,END)

def clear_display_result():
	tab1_display.delete('1.0',END)


# Clear Text  with position 1.0
def clear_text_file():
	displayed_file.delete('1.0',END)

# Clear Result of Functions
def clear_result():
	tab2_display_text.delete('1.0',END)


filename=[]
def openfiles():
	file1 = tk.filedialog.askopenfilename(filetypes=(("Text Files",".txt"),("All files","*"),("csv files",".csv")))
	read_text = open(file1).read()
	filename.append(file1)
	displayed_file.insert(tk.END, read_text)




def get_file_emotion():
	file_name =filename[0]
	large_text = pd.read_csv(file_name)
	def get_emotion_label(text):
		return (emotion(text)[0]['label'])

	large_text['emotion'] = large_text['Text'][0:80].apply(get_emotion_label)

	result = large_text.head(10)
	tab2_display_text.insert(tk.END,result)
	sns.countplot(data=large_text, y='emotion').set_title("Emotion Distribution")
	plt.show()

def get_file_sentiment():
	file_name =filename[0]
	sid = SentimentIntensityAnalyzer()
	df = pd.read_csv(file_name)
	df['scores'] = df['Text'].apply(lambda review: sid.polarity_scores(review))
	df['compound'] = df['scores'].apply(lambda score_dict: score_dict['compound'])
	df['comp_score'] = df['compound'].apply(lambda c: 'positive' if c > 0 else ('negative' if c < 0 else 'Neutral'))
	result = df.head(10)
	tab2_display_text.insert(tk.END, result)
	sns.countplot(data=df, y='comp_score').set_title("Sentiment Distribution")
	plt.show()

def wordcloud():
	file_name = filename[0]
	df = pd.read_csv(file_name)
	text = " ".join(review for review in df.Text)
	word_cloud = WordCloud(collocations=False, background_color='white').generate(text)
	plt.imshow(word_cloud, interpolation='bilinear')
	plt.axis("off")
	plt.show()




# MAIN NLP TAB
l1=Label(tab1,text="Enter Text To Analysis")
l1.grid(row=1,column=0)


raw_entry=StringVar()
entry1=Entry(tab1,textvariable=raw_entry,width=50)
entry1.grid(row=1,column=1)

# bUTTONS
button1=Button(tab1,text="Sentiment", width=12,command=get_sentiment,bg='#f44336',fg='#fff')
button1.grid(row=3,column=0,padx=10,pady=10)
button2=Button(tab1,text="emotion", width=12,command=get_emotion,bg='#f44336',fg='#fff')
button2.grid(row=3,column=1,padx=10,pady=10)




button3=Button(tab1,text="Reset", width=12,command=clear_entry_text,bg="#b9f6ca")
button3.grid(row=3,column=2,padx=10,pady=10)

button4=Button(tab1,text="Clear Result", width=12,command=clear_display_result)
button4.grid(row=4,column=0,padx=10,pady=10)
button5=Button(tab1,text="Close", width=12,command=window.destroy)
button5.grid(row=4,column=1,padx=10,pady=10)
button5=Button(tab1,text="word cloud", width=12,command=word_cloud)
button5.grid(row=4,column=2,padx=10,pady=10)

# Display Screen For Result
tab1_display = Text(tab1)
tab1_display.grid(row=7,column=0, columnspan=3,padx=10,pady=10)

# Allows you to edit
tab1_display.config(state=NORMAL)




# FILE READING  AND PROCESSING TAB
l1=Label(tab2,text="Open File To Process")
l1.grid(row=1,column=1)


displayed_file = ScrolledText(tab2,height=7)# Initial was Text(tab2)
displayed_file.grid(row=2,column=0, columnspan=3,padx=5,pady=3)


# BUTTONS FOR SECOND TAB/FILE READING TAB
b0=Button(tab2,text="Open File", width=12,command=openfiles,bg='#c5cae9')
b0.grid(row=3,column=0,padx=10,pady=10)

b1=Button(tab2,text="Reset ", width=12,command=clear_text_file,bg="#b9f6ca")
b1.grid(row=5,column=0,padx=10,pady=10)

b2=Button(tab2,text="emotion", width=12,command=get_file_emotion,bg='#f44336',fg='#fff')
b2.grid(row=3,column=1,padx=10,pady=10)
b3=Button(tab2,text="sentiment", width=12,command=get_file_sentiment,bg='#f44336',fg='#fff')
b3.grid(row=3,column=2,padx=10,pady=10)


b6=Button(tab2,text="Clear Result", width=12,command=clear_result)
b6.grid(row=5,column=1,padx=10,pady=10)

b7=Button(tab2,text="Close", width=12,command=window.destroy)
b7.grid(row=5,column=2,padx=10,pady=10)
b8=Button(tab2,text="word cloud", width=12,command=wordcloud)
b8.grid(row=5,column=3,padx=10,pady=10)


# Display Screen

# tab2_display_text = Text(tab2)
tab2_display_text = ScrolledText(tab2,height=15,width=125)
tab2_display_text.grid(row=7,column=0, columnspan=3,padx=20,pady=25)

# Allows you to edit
tab2_display_text.config(state=NORMAL)

window.mainloop()