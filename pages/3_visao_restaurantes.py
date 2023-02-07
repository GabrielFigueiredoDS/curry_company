# ==================================
# Liberies 
# ==================================

from haversine import haversine
import plotly.express as px
import pandas as pd
import re
import streamlit as st
from PIL import Image
import folium
from streamlit_folium import folium_static
import numpy as np
import plotly.graph_objects as go

# ==================================
# Funções 
# =================================

def clean_code(df):
    """Esta função tem a responsabilidade de limpar o data frame
    
    Tipos de limpeza:
    1. Remoção dos dados NaN
    2. Mudançã do tipo da coluna de dados
    3. Remoção dos espaços das variáveis de texto
    4. Formatação da coluna de datas
    5. Limpeza da coluna tempo (remoção da variável numérica)
    
    Input: DataFrame
    Output: DataFrame 
    """
    # 1. convertando a coluna Age de testo para numero
    linhas_vazias = df['Delivery_person_Age'] != 'NaN '
    df = df.loc[linhas_vazias, :].copy()

    linhas_vazias = df['Road_traffic_density'] != 'NaN '
    df = df.loc[linhas_vazias, :].copy()

    linhas_vazias = df['City'] != 'NaN '
    df = df.loc[linhas_vazias, :].copy()

    linhas_vazias = df['Festival'] != 'NaN '
    df = df.loc[linhas_vazias, :].copy()

    df['Delivery_person_Age'] = df['Delivery_person_Age'].astype( int )

    # 2. convertendo a coluna Tatings de testo para numero decimal (float)
    df['Delivery_person_Ratings'] = df['Delivery_person_Ratings'].astype( float )

    # 3. convertendo a coluna order_date de testo para data
    df['Order_Date'] = pd.to_datetime( df['Order_Date'], format='%d-%m-%Y' )

    # 4. convertendo multiple_deliveries de testo para numero inteiro (int)
    linhas_vazias = df['multiple_deliveries'] != 'NaN '
    df = df.loc[linhas_vazias, :].copy()
    df['multiple_deliveries'] = df['multiple_deliveries'].astype( int )

    # 6. Removendo os espaços dentro de strings/testos/object
    df.loc[:, 'ID'] = df.loc[:, 'ID'].str.strip()
    df.loc[:, 'Road_traffic_density'] = df.loc[:, 'Road_traffic_density'].str.strip()
    df.loc[:, 'Type_of_order'] = df.loc[:, 'Type_of_order'].str.strip()
    df.loc[:, 'Type_of_vehicle'] = df.loc[:, 'Type_of_vehicle'].str.strip()
    df.loc[:, 'City'] = df.loc[:, 'City'].str.strip()
    df.loc[:, 'Festival'] = df.loc[:, 'Festival'].str.strip()

    # 7. Retirando os numeros da coluna Time_taken(min)
    df['Time_taken(min)'] = df['Time_taken(min)'].apply( lambda x: x.split( '(min) ')[1])
    df['Time_taken(min)'] = df['Time_taken(min)'].astype(int)

    # Remove os NA que forem np.na
    df = df.dropna()
    
    return df

def distance(df):
    """Esta função retorna a distancia média entre os restaurantes
    e o local de entrega. 

    Input: DataFrame
    Outout: Float (valor da distancia média.)
    """
    cols = ['Restaurant_latitude', 'Restaurant_longitude', 'Delivery_location_latitude', 'Delivery_location_longitude']

    # criando colunas distante, que contém as distancias entre o restaurante e local de entrega.
    df['distance'] = df.loc[:, cols].apply(lambda x: haversine(
        (x['Restaurant_latitude'], x['Restaurant_longitude']), 
        (x['Delivery_location_latitude'], x['Delivery_location_longitude'])), axis=1)

    avg_distance = round(df['distance'].mean(), 2)
    return avg_distance

def avg_std_time_delivery(df, festival, op):
    """Esta função calcula o tempo médio e o desvio padrão do tempo de entrega.

    Parâmetros:
    Input:
     - df: DataFrame com os dados necessários para o cálculo.
     - op: Tipo da operação que precisa ser calculado.
             'avg_time: Calcula o tempo médio.
             ''std_time: Calcula o desvio padrão do tempo
     - festival: Selecionando a coluna festival.
             'Yes': Retorna a operação escolhida que tiveram festival.
             'No': Retorna a operação escolhida que não tiveram festival.
    """
    cols = ['Time_taken(min)', 'Festival']
    df_grouped = (df.loc[:, cols]
              .groupby(['Festival'])
              .agg({'Time_taken(min)': ['mean', 'std']}))

    df_grouped.columns = ['avg_time', 'std_time']
    df_grouped = df_grouped.reset_index()
    df_grouped = round(df_grouped.loc[df_grouped['Festival'] == festival, op], 2)

    return df_grouped

def std_bar_plotly(df):
    """Esta função desenha um gráfico de barra com desvio padrão """
    cols = ['City', 'Time_taken(min)']

    df_grouped = df.loc[:, cols].groupby(['City']).agg({'Time_taken(min)': ['mean', 'std']})

    df_grouped.columns = ['avg_time', 'std_time']

    df_grouped = df_grouped.reset_index()

    fig = go.Figure()
    fig.add_trace(go.Bar (name='Control',
                         x=df_grouped['City'],
                         y=df_grouped['avg_time'],
                         error_y=dict(type='data', array=df_grouped['std_time'])))
    return fig

def avg_std_time_graph(df):
    """Esta função desenha um gráfico de pizza que retorna o tempo 
    médio de entrega por cidade """

    # criando colunas distante, que contém as distancias entre o restaurante e local de entrega.
    df['distance'] = df.loc[:, ['Restaurant_latitude', 'Restaurant_longitude', 'Delivery_location_latitude', 'Delivery_location_longitude']].apply(lambda x: haversine(
        (x['Restaurant_latitude'], x['Restaurant_longitude']), 
        (x['Delivery_location_latitude'], x['Delivery_location_longitude'])), axis=1)

    avg_distance = df.loc[:, ['City', 'distance']].groupby('City').mean().reset_index()
    fig = go.Figure(data=[go.Pie(labels=avg_distance['City'], values=avg_distance['distance'], pull=[0, 0.1, 0])])

    return fig

def sunburst_plotly(df):
    """Esta função desenha um gráfico de sol. Chamado sunburst"""
    
    cols = ['City', 'Time_taken(min)', 'Road_traffic_density']
    df_grouped = df.loc[:, cols].groupby( ['City', 'Road_traffic_density']).agg({'Time_taken(min)':['mean', 'std']})
    df_grouped.columns = ['avg_time', 'std_time']
    df_grouped = df_grouped.reset_index()
    fig = px.sunburst(df_grouped, path=['City', 'Road_traffic_density'], values='avg_time',
                      color='std_time', color_continuous_scale='RdBu',
                      color_continuous_midpoint=np.average(df_grouped['std_time']))
    
    return fig

def table_mean_std(df):
    """Esta função retorna um DataFrame que contém a média e o desvio padrão 
    por cidade e tipo de pedido. 
    """
    # O tempo médio e o desvio padrão de entrega por cidade e tipo de pedido.
    cols = ['City', 'Time_taken(min)', 'Type_of_order']
    df_grouped = df.loc[:, cols].groupby(['City', 'Type_of_order']).agg({'Time_taken(min)': ['mean', 'std']})
    df_grouped.columns = ['avg_time', 'std_time']
    df_grouped = df_grouped.reset_index()

    return df_grouped
# ----------------------- Inicio da Estrutura lógica do código-------------------#

# ==================================
# Configurações Página 
# ==================================
st.set_page_config(page_title='Visão Restaurantes', page_icon="🍜", layout='wide')

# ==================================
# DataSet 
# ==================================
df_raw = pd.read_csv('dataset/train.csv')

# Fazendo uma cópia do DataFrame Lido
df = df_raw.copy()

# ==================================
# Limpeza dos dados 
# ==================================
df = clean_code( df )

# ==================================
# Barra Lateral
# ==================================

st.header('Marketplace - Visão Restaurantes')

image_path = 'logo.png'
image = Image.open(image_path)
st.sidebar.image(image, width=220)

st.sidebar.markdown('# Cury Company')
st.sidebar.markdown('## Fastest Delivery in Town')
st.sidebar.markdown("""---""")

st.sidebar.markdown('## Selecione uma data limite')

date_slider = st.sidebar.slider(
    'Até qual valor?',
    value=pd.datetime(2022,4,13),
    min_value=pd.datetime( 2022, 2, 11),
    max_value=pd.datetime( 2022, 4, 6),
    format='DD-MM-YYYY')

st.header( date_slider )
st.sidebar.markdown("""---""")

traffic_options = st.sidebar.multiselect(
    'Quais as condições do trânsito',
    ['Low', 'Medium', 'High', 'Jam'],
    default=['Low', 'Medium', 'High', 'Jam'])

st.sidebar.markdown("""---""")
st.sidebar.markdown('### Powered by Gabriel Figueirêdo')

# Filtro data
linhas_selecionadas = df['Order_Date'] < date_slider
df = df.loc[linhas_selecionadas, :]

# Filtro transito
linhas_selecionadas = df['Road_traffic_density'].isin(traffic_options)
df = df.loc[linhas_selecionadas, :]

# ==================================
# Layout no Streamlit
# ==================================
tab1, tab2, tab3 = st.tabs(['Visão Gerencial', '_', '_'])

with tab1:
    with st.container():
        st.title('Overal Metrics')
        
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        with col1:
            delivery_unique = len(df['Delivery_person_ID'].unique())
            col1.metric('Entregadores únicos', delivery_unique)
            
        with col2:
            avg_distance = distance(df)
            col2.metric('Distancia Média', avg_distance)

        with col3:
            df_grouped = avg_std_time_delivery(df, 'Yes', 'avg_time')
            col3.metric('Tempo Médio de Entrega', df_grouped)
            
        with col4:
            df_grouped = avg_std_time_delivery(df, 'Yes', 'std_time')
            col4.metric('STD Entrega', df_grouped)

        with col5:
            df_grouped = avg_std_time_delivery(df, 'No', 'avg_time')
            col5.metric('Tempo Médio', df_grouped)
            
        with col6:
            df_grouped = avg_std_time_delivery(df, 'No', 'std_time')
            col6.metric('Tempo Médio', df_grouped)

    with st.container():
        st.markdown('''---''')
        col1, col2 = st.columns(2)
        
        with col1:          
            fig = std_bar_plotly(df)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:     
            df_grouped = table_mean_std(df)
            st.dataframe(df_grouped)
                
    with st.container():
        st.markdown('''---''')
        st.title('Distribuição do Tempo')
        
        col1, col2 = st.columns( 2 )
        
        with col1:
            fig = avg_std_time_graph(df)
            st.plotly_chart(fig, use_container_width=True)
                      
        with col2:
            fig = sunburst_plotly(df)
            st.plotly_chart(fig, use_container_width=True)        
            