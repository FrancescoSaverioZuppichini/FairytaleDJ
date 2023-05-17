ChatGPT + LangChain + DeepLake + Disney + Streamlit = FairytaleDJ 🎵🏰🔮

I love Disney songs, as you may know, so I am sharing an app that will suggest Disney songs based on user input, give it a try! 

The demo is on Hugging Face spaces built with Streamlit 🤗 🚀 : https://huggingface.co/spaces/Francesco/FairytaleDJ

(thanks to Freddy Boulton for the suggestion)

I had embedded using gpt-3 around 100 Disney songs with the following strategy:

▪️ for each song, ask chatGPT to give me 8 emotions based on the lyrics
▪️ embed each song using these emotions with gpt3 on Activeloop DeepLake vector db (I've collab with them for this one)
▪️ when a user inputs something, use chatGPT to convert it to a list of emotions
▪️ do a similarity search on the vector db with the user's emotions
▪️ filter out low-scoring songs
▪️ Sample n songs based on their final score

GitHub: https://github.com/FrancescoSaverioZuppichini/FairytaleDJ/blob/main/app.py
Explanation Video: https://www.youtube.com/watch?v=nJl0LesTxzs

#ai #ml #chatgpt #llms #opensource #machinelearning #programming #deeplearning 