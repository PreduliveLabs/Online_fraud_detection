import numpy as np
import joblib
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer





RFC=joblib.load("RFC.sav")

model0=joblib.load("model0.sav")
model1=joblib.load("model1.sav")
model2=joblib.load("model2.sav")
model3=joblib.load("model3.sav")
model4=joblib.load("model4.sav")
model5=joblib.load("model5.sav")
model6=joblib.load("model6.sav")
model7=joblib.load("model7.sav")
model8=joblib.load("model8.sav")
model9=joblib.load("model9.sav")
model10=joblib.load("model10.sav")
model11=joblib.load("model11.sav")









from sklearn.feature_extraction.text import HashingVectorizer,TfidfVectorizer

from flask import Flask, render_template, request, url_for,send_from_directory,Response
import re
port_stem = PorterStemmer()

def clean_data(combine):
    stemmed_content = re.sub('[^a-zA-Z)]',' ', combine)
    stemmed_content = stemmed_content.lower()
    stemmed_content = stemmed_content.split()
    stemmed_content = [port_stem.stem(word) for word in stemmed_content if not word in stopwords.words('english')]
    stemmed_content = ' '.join(stemmed_content)
    return stemmed_content




d={9: 'Online and Social Media Related Crime',
 7: 'Online Financial Fraud',
 8: 'Online Gambling  Betting',
 0: 'Any Other Cyber Crime',
 3: 'Cyber Attack/ Dependent Crimes',
 2: 'Cryptocurrency Crime',
 5: 'Hacking  Damage to computercomputer system etc',
 4: 'Cyber Terrorism',
 6: 'Online Cyber Trafficking',
 10: 'Ransomware',
 11: 'Report Unlawful Content',
 1: 'Crime Against Women & Children'}

d0=['Other']
d1=['Computer Generated CSAM/CSEM', 'Cyber Blackmailing & Threatening',
       'Sexual Harassment']
d2=['Cryptocurrency Fraud']
d3=['Data Breach/Theft',
       'Denial of Service (DoS)/Distributed Denial of Service (DDOS) attacks',
       'Malware Attack', 'Hacking/Defacement', 'SQL Injection',
       'Ransomware Attack', 'Tampering with computer source documents']
d4=['Cyber Terrorism']
d5=['Email Hacking', 'Unauthorised AccessData Breach',
       'Website DefacementHacking',
       'Damage to computer computer systems etc',
       'Tampering with computer source documents']


d6=['Online Trafficking']
d7=['Fraud CallVishing', 'UPI Related Frauds',
       'Internet Banking Related Fraud',
       'DebitCredit Card FraudSim Swap Fraud', 'EWallet Related Fraud',
       'Business Email CompromiseEmail Takeover', 'DematDepository Fraud']
d8=['Online Gambling  Betting']
d9=['Cyber Bullying  Stalking  Sexting', 'Online Job Fraud',
       'Profile Hacking Identity Theft', 'Cheating by Impersonation',
       'FakeImpersonating Profile',
       'Provocative Speech for unlawful acts', 'Online Matrimonial Fraud',
       'Impersonating Email', 'EMail Phishing', 'Intimidating Email']
d10=['Ransomware']
d11=['Against Interest of sovereignty or integrity of India']

list1=[d0,d1,d2,d3,d4,d5,d6,d7,d8,d9,d10,d11]

app = Flask(__name__)
app.config["SECRET_KEY"] = 'ajashjkjm'

@app.route('/')
def home():
    return render_template('main_page.html')



@app.route('/file', methods=['GET', 'POST'])
def main_page():
    if request.method == 'POST':
        text=request.form["text"]
        if text.strip() == "" :
            return render_template('main_page.html', prediction=None,msg="text field required")
        else:
            text=text.strip()
            text=clean_data(text)
            tfv = HashingVectorizer(n_features=100, norm=None, alternate_sign=False, stop_words='english')
            values = tfv.fit_transform([text]).toarray()
            prediction=RFC.predict(values)
            print(prediction)
            if(prediction[0]==0):
                prediction1=model0.predict(values)
            elif(prediction[0]==1):
                prediction1 = model1.predict(values)
            elif(prediction[0]==2):
                prediction1 = model2.predict(values)

            elif (prediction[0] == 3):
                prediction1 = model3.predict(values)

            elif (prediction[0] == 4):
                prediction1 = model4.predict(values)

            elif (prediction[0] == 5):
                prediction1 = model5.predict(values)


            elif (prediction[0] == 6):
                prediction1 = model6.predict(values)

            elif (prediction[0] == 7):
                prediction1 = model7.predict(values)

            elif (prediction[0] == 8):
                prediction1 = model8.predict(values)

            elif (prediction[0] == 9):
                prediction1 = model9.predict(values)

            elif (prediction[0] == 10):
                prediction1 = model10.predict(values)

            elif (prediction[0] == 11):
                prediction1 = model11.predict(values)


        return render_template('main_page.html', prediction=d[prediction[0]],prediction1=list1[prediction[0]][prediction1[0]],text=text)

    return render_template('main_page.html')








if __name__ == '__main__':
    app.run(debug=True)

