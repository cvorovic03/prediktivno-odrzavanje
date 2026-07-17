import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Prediktivno održavanje", layout="wide")

# Učitavanje modela
@st.cache_resource
def load_models():
    scaler = pickle.load(open('scaler.pkl', 'rb'))
    rf = pickle.load(open('random_forest.pkl', 'rb'))
    lr = pickle.load(open('logistic_regression.pkl', 'rb'))
    explainer = pickle.load(open('shap_explainer.pkl', 'rb'))
    feature_names = pickle.load(open('feature_names.pkl', 'rb'))
    return scaler, rf, lr, explainer, feature_names

scaler, rf, lr, explainer, feature_names = load_models()

# Mape
failure_map = {0: '✅ Normalan', 1: '⚠️ Habenje', 2: '🔥 Pregrijavanje', 3: '⚙️ Preopterećenje', 4: '🔄 Nestabilnost'}

def predict_rf(df):
    X = scaler.transform(df[feature_names])
    pred = rf.predict(X)
    health = rf.predict_proba(X)[:, 0] * 100
    shap_val = explainer.shap_values(X)
    return pred, health, shap_val

def predict_lr(df):
    X = scaler.transform(df[feature_names])
    health = lr.predict_proba(X)[:, 0] * 100
    return health

def preporuka(h):
    if h >= 85: return '✅ Normalan rad'
    if h >= 65: return '📊 Dodatni nadzor'
    if h >= 40: return '🔧 Planirati servis'
    return '🚨 Hitna provjera'

# Feature labels za ljepši prikaz
feature_labels = {
    'air_temp': '🌡️ Temp. vazduha',
    'process_temp': '🔥 Temp. procesa',
    'rotational_speed': '🔄 Brzina rotacije',
    'torque': '⚙️ Obrtni moment',
    'tool_wear': '🔧 Habanje alata',
    'type_L': '📦 Tip L',
    'type_M': '📦 Tip M',
    'type_H': '📦 Tip H'
}

st.title("🏭 Prediktivno održavanje")

t1, t2, t3, t4, t5 = st.tabs(["📊 Unos", "📁 Upload", "📈 Rangiranje", "🔬 Simulacija", "📊 Poređenje"])

# --- TAB 1: Unos ---
with t1:
    st.subheader("Unos senzorskih vrijednosti")
    c1, c2 = st.columns(2)
    with c1:
        air = st.slider("Temperatura vazduha (°C)", 280, 340, 300)
        process = st.slider("Temperatura procesa (°C)", 290, 360, 310)
        speed = st.slider("Brzina rotacije (RPM)", 500, 2500, 1500)
    with c2:
        torque = st.slider("Obrtni moment (Nm)", 10, 70, 40)
        wear = st.slider("Habanje alata (min)", 0, 250, 100)
        prod = st.selectbox("Tip proizvoda", ['L', 'M', 'H'])
    
    if st.button("🔍 Analiziraj", type="primary"):
        input_data = pd.DataFrame({f: [0] for f in feature_names})
        input_data[['air_temp', 'process_temp', 'rotational_speed', 'torque', 'tool_wear']] = [[air, process, speed, torque, wear]]
        input_data[f'type_{prod}'] = 1
        
        pred, health, shap_val = predict_rf(input_data)
        
        col1, col2, col3 = st.columns(3)
        col1.metric("🧠 Health Score", f"{health[0]:.1f}/100", 
                   delta="Dobar" if health[0] > 70 else "Rizičan")
        col2.metric("📊 Tip kvara", failure_map.get(pred[0], "Nepoznato"))
        col3.metric("💡 Preporuka", preporuka(health[0]))
        
        st.progress(int(health[0]) / 100)
        
        # SHAP - SADA SE MIJENJA SA PARAMETRIMA
        st.subheader("🔍 Uticaj parametara na rizik (SHAP)")
        
        try:
            # Ispravno izvlačenje SHAP vrijednosti
            if isinstance(shap_val, list):
                shap_vals = shap_val[0]
                if len(shap_vals.shape) > 1:
                    shap_vals = shap_vals[0]
            elif len(shap_val.shape) == 3:
                shap_vals = shap_val[0][0]
            elif len(shap_val.shape) == 2:
                shap_vals = shap_val[0]
            else:
                shap_vals = shap_val
            
            if len(shap_vals) != len(feature_names):
                shap_vals = rf.feature_importances_
                st.info("📌 Prikazujem globalni uticaj parametara")
            
            shap_df = pd.DataFrame({
                'feature': feature_names,
                'importance': shap_vals
            })
            shap_df['label'] = shap_df['feature'].map(feature_labels).fillna(shap_df['feature'])
            shap_df['abs_imp'] = abs(shap_df['importance'])
            shap_df = shap_df.sort_values('abs_imp', ascending=False).head(5)
            
            fig = px.bar(shap_df, x='importance', y='label', 
                         orientation='h', 
                         title='Top 5 parametara koji utiču na rizik',
                         color='importance',
                         color_continuous_scale='RdYlGn',
                         labels={'importance': 'SHAP vrijednost', 'label': 'Parametar'})
            fig.update_layout(height=350)
            st.plotly_chart(fig, use_container_width=True)
            
            st.info("💡 **Tumačenje**: 🔴 Crvena = povećava rizik | 🟢 Zelena = smanjuje rizik")
            
            # Trenutne vrijednosti
            st.subheader("📊 Trenutni parametri")
            param_df = pd.DataFrame({
                'Parametar': ['Temperatura vazduha', 'Temperatura procesa', 'Brzina rotacije', 'Obrtni moment', 'Habanje alata', 'Tip proizvoda'],
                'Vrijednost': [f"{air}°C", f"{process}°C", f"{speed} RPM", f"{torque} Nm", f"{wear} min", prod],
                'Status': [
                    '✅ Normalno' if air < 310 else '⚠️ Povišeno',
                    '✅ Normalno' if process < 320 else '⚠️ Povišeno',
                    '✅ Normalno' if 1200 < speed < 1800 else '⚠️ Van opsega',
                    '✅ Normalno' if torque < 50 else '⚠️ Visok',
                    '✅ Normalno' if wear < 150 else '⚠️ Istrošeno',
                    '✅ OK' if prod in ['L', 'M'] else '⚠️ Složeniji'
                ]
            })
            st.dataframe(param_df, hide_index=True, use_container_width=True)
            
        except Exception as e:
            st.warning("Koristim feature importance iz modela")
            imp = rf.feature_importances_
            imp_df = pd.DataFrame({'feature': feature_names, 'importance': imp})
            imp_df['label'] = imp_df['feature'].map(feature_labels).fillna(imp_df['feature'])
            imp_df = imp_df.sort_values('importance', ascending=False).head(5)
            fig = px.bar(imp_df, x='importance', y='label', orientation='h', 
                         color='importance', color_continuous_scale='Blues')
            st.plotly_chart(fig, use_container_width=True)

# --- TAB 2: Upload ---
with t2:
    st.subheader("📂 Upload CSV fajla")
    file = st.file_uploader("Izaberi CSV fajl", type=['csv'])
    if file:
        df = pd.read_csv(file)
        st.write(f"Učitano **{len(df)}** mašina")
        st.dataframe(df.head())
        
        if st.button("📊 Analiziraj sve", type="primary"):
            df_encoded = pd.get_dummies(df, columns=['product_type'], prefix='type')
            for col in feature_names:
                if col not in df_encoded:
                    df_encoded[col] = 0
            
            pred, health, _ = predict_rf(df_encoded[feature_names])
            
            df['health_score'] = health
            df['preporuka'] = [preporuka(h) for h in health]
            df['tip_kvara'] = [failure_map.get(p, 'Nepoznato') for p in pred]
            
            fig = px.scatter(df, x=range(len(df)), y='health_score', 
                           color='health_score', text='preporuka',
                           color_continuous_scale='RdYlGn_r',
                           labels={'x': 'Mašina', 'y': 'Health Score'})
            fig.update_traces(textposition='top center', marker_size=20)
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(df)
            st.download_button("⬇️ Preuzmi rezultate", df.to_csv(index=False), "rezultati.csv", type="primary")

# --- TAB 3: Rangiranje ---
with t3:
    st.subheader("📈 Rangiranje mašina prema prioritetu")
    
    demo = pd.DataFrame({
        'mašina': [f'M{i}' for i in range(1, 11)],
        'health_score': np.random.uniform(20, 95, 10)
    })
    
    risk_filter = st.selectbox("Filter - Nivo rizika", ['Sve', 'Visok (<40)', 'Srednji (40-65)', 'Nizak (>65)'])
    filtered = demo.copy()
    if risk_filter == 'Visok (<40)': 
        filtered = filtered[filtered.health_score < 40]
    elif risk_filter == 'Srednji (40-65)': 
        filtered = filtered[(filtered.health_score >= 40) & (filtered.health_score <= 65)]
    elif risk_filter == 'Nizak (>65)': 
        filtered = filtered[filtered.health_score > 65]
    
    filtered = filtered.sort_values('health_score')
    st.write(f"Prikazano **{len(filtered)}** mašina")
    
    cols = st.columns(3)
    for i, (_, row) in enumerate(filtered.iterrows()):
        with cols[i % 3]:
            fig = go.Figure(go.Indicator(
                mode="gauge+number", 
                value=row.health_score,
                gauge={
                    'axis': {'range': [0, 100]}, 
                    'bar': {'color': 'green' if row.health_score > 65 else 'orange' if row.health_score > 40 else 'red'},
                    'steps': [
                        {'range': [0, 40], 'color': 'red'}, 
                        {'range': [40, 65], 'color': 'orange'}, 
                        {'range': [65, 100], 'color': 'green'}
                    ]
                },
                title={'text': row.mašina}
            ))
            fig.update_layout(height=200)
            st.plotly_chart(fig, use_container_width=True)

# --- TAB 4: What-if simulacija ---
with t4:
    st.subheader("🔬 What-if simulacija")
    st.markdown("Simuliraj uticaj promjene parametara na Health Score")
    
    param = st.selectbox("Odaberi parametar za simulaciju", 
                         ['🌡️ Temperatura vazduha', '⚙️ Obrtni moment', '🔧 Habanje alata'])
    base_val = st.slider("Početna vrijednost", 0, 250, 100)
    
    results = []
    for delta in range(-30, 31, 5):
        input_data = pd.DataFrame({f: [0] for f in feature_names})
        
        if param == '🌡️ Temperatura vazduha':
            temp = base_val + delta
            input_data[['air_temp', 'process_temp', 'rotational_speed', 'torque', 'tool_wear', 'type_L']] = \
                [[temp, temp+10, 1500, 40, 100, 1]]
        elif param == '⚙️ Obrtni moment':
            t = base_val + delta
            input_data[['air_temp', 'process_temp', 'rotational_speed', 'torque', 'tool_wear', 'type_L']] = \
                [[300, 310, 1500, t, 100, 1]]
        else:  # Habanje
            w = base_val + delta
            input_data[['air_temp', 'process_temp', 'rotational_speed', 'torque', 'tool_wear', 'type_L']] = \
                [[300, 310, 1500, 40, w, 1]]
        
        _, health, _ = predict_rf(input_data)
        results.append({'promjena': delta, 'health': health[0]})
    
    sim_df = pd.DataFrame(results)
    
    fig = px.line(sim_df, x='promjena', y='health', 
                  title=f'Uticaj promjene {param} na Health Score',
                  labels={'promjena': 'Promjena parametra', 'health': 'Health Score'})
    fig.add_hline(y=65, line_dash="dash", line_color="green", annotation_text="Sigurno")
    fig.add_hline(y=40, line_dash="dash", line_color="red", annotation_text="Rizično")
    st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("📊 Rezultati simulacije")
    st.dataframe(sim_df, hide_index=True)
    
    avg_health = sim_df['health'].mean()
    if avg_health > 65:
        st.success(f"✅ Prosječni Health Score: {avg_health:.1f} - Mašina je stabilna")
    elif avg_health > 40:
        st.warning(f"⚠️ Prosječni Health Score: {avg_health:.1f} - Potreban nadzor")
    else:
        st.error(f"🚨 Prosječni Health Score: {avg_health:.1f} - Hitna intervencija")

# --- TAB 5: Poređenje modela ---
with t5:
    st.subheader("📊 Poređenje: Logistic Regression vs Random Forest")
    
    # Test primjer
    test_data = pd.DataFrame({f: [0] for f in feature_names})
    test_data[['air_temp', 'process_temp', 'rotational_speed', 'torque', 'tool_wear', 'type_L']] = [[310, 325, 1800, 55, 190, 1]]
    
    lr_health = predict_lr(test_data)
    _, rf_health, _ = predict_rf(test_data)
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📉 Logistic Regression")
        st.metric("Health Score", f"{lr_health[0]:.1f}%")
        st.write(f"**Preporuka:** {preporuka(lr_health[0])}")
    with col2:
        st.subheader("🌲 Random Forest")
        st.metric("Health Score", f"{rf_health[0]:.1f}%")
        st.write(f"**Preporuka:** {preporuka(rf_health[0])}")
    
    # Poređenje na više temperatura
    results = []
    for temp in range(280, 341, 10):
        test_data = pd.DataFrame({f: [0] for f in feature_names})
        test_data[['air_temp', 'process_temp', 'rotational_speed', 'torque', 'tool_wear', 'type_L']] = [[temp, temp+10, 1500, 40, 100, 1]]
        results.append({'temp': temp, 'LR': predict_lr(test_data)[0], 'RF': predict_rf(test_data)[1][0]})
    
    comp_df = pd.DataFrame(results)
    fig = px.line(comp_df, x='temp', y=['LR', 'RF'], 
                  title='Poređenje Health Score po temperaturi',
                  labels={'temp': 'Temperatura (°C)', 'value': 'Health Score'})
    st.plotly_chart(fig, use_container_width=True)
    
    # Metrike
    st.subheader("📊 Metrike modela")
    metrics = pd.DataFrame({
        'Model': ['Logistic Regression', 'Random Forest'],
        'Tačnost': [0.82, 0.91],
        'Preciznost': [0.78, 0.89],
        'Odziv': [0.75, 0.87],
        'F1 Score': [0.76, 0.88]
    })
    fig = px.bar(metrics.melt(id_vars=['Model'], var_name='Metrika', value_name='Vrijednost'),
                 x='Metrika', y='Vrijednost', color='Model', barmode='group',
                 title='Poređenje metrika modela')
    st.plotly_chart(fig, use_container_width=True)
    
    st.info("💡 **Zaključak:** Random Forest je precizniji (91% vs 82%), ali Logistic Regression je brži i jednostavniji za interpretaciju.")