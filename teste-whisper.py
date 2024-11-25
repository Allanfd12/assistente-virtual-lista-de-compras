import whisper
model = whisper.load_model("base")
result = model.transcribe("C:/Users/allan/Documents/GIT/AULA IA/aula 5/temp/1731969151.1008244-se que nãmose fosse um boltouelinca mernoe nunca mei valeme a sefado gracet.wav")
print(f' The text in video: \n {result["text"]}')