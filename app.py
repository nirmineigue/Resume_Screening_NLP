import nltk
import re
import streamlit as st
import pickle

#from sklearn.feature_extraction.text import TfidfVectorizer
#tfidf = TfidfVectorizer(stop_words= 'english')

nltk.download('puntk')
nltk.download('stopwors')

# Loading models 
clf = pickle.load(open('clf.pkl', 'rb'))
tfidfd  = pickle.load(open ('tfidf.pkl', 'rb'))

def cleanResume(txt):
    cleanTxt = re.sub('http\S+\s', ' ' ,txt)
    cleanTxt = re.sub('@\S+', ' ' ,cleanTxt)
    cleanTxt = re.sub('#\S+', ' ' ,cleanTxt)
    cleanTxt = re.sub('RT|cc', ' ' ,cleanTxt)
    cleanTxt = re.sub('[%s]' % re.escape("""!"$%&'()*+,-./:;<=>?@'[\]^_{|}~"""), ' ' , cleanTxt)
    cleanTxt = re.sub(r'[^\x00-\x7f]', ' ',cleanTxt)
    cleanTxt = re.sub('\s+', ' ' ,cleanTxt)
    return cleanTxt

# web app
def main():
    st.title("Resume Screening App")
    upload_file = st.file_uploader('Upload Resume',  type = ['txt', 'pdf'])

    if upload_file is not None:
        try:
            resume_bytes = upload_file.read()
            resume_text = resume_bytes.decode('utf-8')
        except UnicodeDecodeError:
            # if UTF-8 decoding fails, try decoding with 'latin-1'
            resume_text = resume_bytes.decode('latin-1')
        
        
        cleaned_resume = cleanResume(resume_text)
        input_features = tfidfd.transform([cleaned_resume])
        prediction_id = clf.predict(input_features)[0]
        st.write(prediction_id)
        category_mapping = {
            15 : "Java Developer",
            23 : "Testing",
            8 : "Devops Engineer",
            20 : " Python Developer",
            24 : "Web Designer",
            12 : "HR",
            13: "Hadoop",
            3: "Blockchain",
            10: "ETL Developer",
            18: "Operations Manager",
            6: "Data Science",
            22: "Sales",
            16: "Mechanical Engineer",
            1: "Arts",
            7: "Database",
            11: "Electrical Engineering",
            14: "Health and fitness",
            19: "PMO",
            4: "Business Analyst",
            9: "DotNet Developer",
            2: "Automation Testing",
            17: "Network Security Engineer",
            21: "SAP Developer",
            5: "Civil Engineer",
            0: "Advocate",
        }
        category_name = category_mapping.get(prediction_id, "Unknown")

        st.write("Predicted Category:", category_name)




# python main
if __name__ == "__main__":
    main()
