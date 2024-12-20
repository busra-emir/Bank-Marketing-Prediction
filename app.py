import streamlit as st
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Model ve scaler'ları yükleme
@st.cache_resource
def load_model():
    try:
        model = joblib.load('BSA_model.pkl')
        return model
    except Exception as e:
        st.error(f"Model yüklenirken hata oluştu: {e}")
        return None

def prepare_input_data(df):
    """Veri ön işleme adımlarını gerçekleştirir"""
    # Kategorik değişkenler için mapping
    job_mapping = {
        'admin.': 0, 'blue-collar': 1, 'entrepreneur': 2, 'housemaid': 3,
        'management': 4, 'retired': 5, 'self-employed': 6, 'services': 7,
        'student': 8, 'technician': 9, 'unemployed': 10, 'unknown': 11
    }
    
    marital_mapping = {'divorced': 0, 'married': 1, 'single': 2}
    
    education_mapping = {
        'basic.4y': 0, 'basic.6y': 1, 'basic.9y': 2, 'high.school': 3,
        'illiterate': 4, 'professional.course': 5, 'university.degree': 6, 'unknown': 7
    }
    
    default_mapping = {'no': 0, 'yes': 1}
    housing_mapping = {'no': 0, 'yes': 1}
    loan_mapping = {'no': 0, 'yes': 1}
    contact_mapping = {'cellular': 0, 'telephone': 1, 'unknown': 2}
    
    month_mapping = {
        'jan': 0, 'feb': 1, 'mar': 2, 'apr': 3, 'may': 4, 'jun': 5,
        'jul': 6, 'aug': 7, 'sep': 8, 'oct': 9, 'nov': 10, 'dec': 11
    }
    
    day_mapping = {'mon': 0, 'tue': 1, 'wed': 2, 'thu': 3, 'fri': 4}
    poutcome_mapping = {'failure': 0, 'nonexistent': 1, 'success': 2, 'unknown': 3}

    # Kategorik değişkenleri dönüştür
    df['job'] = df['job'].map(job_mapping)
    df['marital'] = df['marital'].map(marital_mapping)
    df['education'] = df['education'].map(education_mapping)
    df['default'] = df['default'].map(default_mapping)
    df['housing'] = df['housing'].map(housing_mapping)
    df['loan'] = df['loan'].map(loan_mapping)
    df['contact'] = df['contact'].map(contact_mapping)
    df['month'] = df['month'].map(month_mapping)
    df['day_of_week'] = df['day_of_week'].map(day_mapping)
    df['poutcome'] = df['poutcome'].map(poutcome_mapping)

    # Sayısal değişkenler için scaler
    scaler = StandardScaler()
    numerical_cols = ['age', 'campaign', 'pdays', 'previous', 'emp.var.rate',
                     'cons.price.idx', 'cons.conf.idx', 'euribor3m', 'nr.employed']
    
    # Sayısal değişkenleri normalize et
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
    
    return df

def main():
    st.title('Bank Marketing Prediction App')
    
    # Load model
    model = load_model()
    if model is None:
        return
    
    # Input fields
    st.subheader('Müşteri Bilgileri')
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.number_input('Yaş', min_value=18, max_value=100, value=30)
        job = st.selectbox('Meslek', options=['admin.', 'blue-collar', 'entrepreneur', 'housemaid', 
                                            'management', 'retired', 'self-employed', 'services', 
                                            'student', 'technician', 'unemployed', 'unknown'])
        marital = st.selectbox('Medeni Durum', options=['married', 'divorced', 'single'])
        education = st.selectbox('Eğitim', options=['basic.4y', 'basic.6y', 'basic.9y', 'high.school',
                                                  'illiterate', 'professional.course', 'university.degree', 
                                                  'unknown'])
        default = st.selectbox('Kredi Temerrüdü', options=['no', 'yes'])
        housing = st.selectbox('Konut Kredisi', options=['no', 'yes'])
        loan = st.selectbox('Kişisel Kredi', options=['no', 'yes'])
        
    with col2:
        contact = st.selectbox('İletişim Türü', options=['cellular', 'telephone', 'unknown'])
        month = st.selectbox('Ay', options=['jan', 'feb', 'mar', 'apr', 'may', 'jun',
                                          'jul', 'aug', 'sep', 'oct', 'nov', 'dec'])
        day_of_week = st.selectbox('Haftanın Günü', options=['mon', 'tue', 'wed', 'thu', 'fri'])
        campaign = st.number_input('Kampanya İletişim Sayısı', min_value=1, max_value=50, value=1)
        pdays = st.number_input('Son İletişimden Bu Yana Geçen Gün', min_value=0, value=0)
        previous = st.number_input('Önceki Kampanya İletişim Sayısı', min_value=0, value=0)
        poutcome = st.selectbox('Önceki Kampanya Sonucu', options=['failure', 'nonexistent', 'success', 'unknown'])
    
    st.subheader('Ekonomik Göstergeler')
    col3, col4 = st.columns(2)
    
    with col3:
        emp_var_rate = st.number_input('İstihdam Değişim Oranı', value=0.0, step=0.1)
        cons_price_idx = st.number_input('Tüketici Fiyat Endeksi', value=93.2, step=0.1)
        cons_conf_idx = st.number_input('Tüketici Güven Endeksi', value=-36.4, step=0.1)
    
    with col4:
        euribor3m = st.number_input('Euribor 3 Aylık Oran', value=4.857, step=0.001)
        nr_employed = st.number_input('Çalışan Sayısı', value=5191.0, step=1.0)

    # Predict butonu
    if st.button('Tahmin Et'):
        try:
            # Veriyi DataFrame'e dönüştür
            input_data = pd.DataFrame({
                'age': [age],
                'job': [job],
                'marital': [marital],
                'education': [education],
                'default': [default],
                'housing': [housing],
                'loan': [loan],
                'contact': [contact],
                'month': [month],
                'day_of_week': [day_of_week],
                'campaign': [campaign],
                'pdays': [pdays],
                'previous': [previous],
                'poutcome': [poutcome],
                'emp.var.rate': [emp_var_rate],
                'cons.price.idx': [cons_price_idx],
                'cons.conf.idx': [cons_conf_idx],
                'euribor3m': [euribor3m],
                'nr.employed': [nr_employed]
            })
            
            # Veri ön işleme
            processed_data = prepare_input_data(input_data)
            
            # Tahmin
            prediction = model.predict(processed_data)
            probability = model.predict_proba(processed_data)
            
            # Sonuçları göster
            if prediction[0] == 1:
                st.success('Tahmin: Müşteri vadeli mevduat ürününü satın alma olasılığı YÜKSEK')
            else:
                st.error('Tahmin: Müşteri vadeli mevduat ürününü satın alma olasılığı DÜŞÜK')
                
            st.write(f'Satın alma olasılığı: {probability[0][1]:.2%}')
            
            # Debug bilgisi (geliştirme aşamasında yararlı olabilir)
            if st.checkbox('Debug bilgisini göster'):
                st.write("İşlenmiş veri:")
                st.write(processed_data)
                
        except Exception as e:
            st.error(f"Tahmin yapılırken bir hata oluştu: {e}")

if __name__ == '__main__':
    main()
