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

# ==================================
# Funções 
# ==================================

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
    # Remover spaco da string
    df['ID'] = df['ID'].str.strip()
    df['Delivery_person_ID'] = df['Delivery_person_ID'].str.strip()

    # Excluir as linhas com a idade dos entregadores vazia
    # ( Conceitos de seleção condicional )
    linhas_vazias = df['Delivery_person_Age'] != 'NaN '
    df = df.loc[linhas_vazias, :]

    # Conversao de texto/categoria/string para numeros inteiros
    df['Delivery_person_Age'] = df['Delivery_person_Age'].astype( int )

    # Conversao de texto/categoria/strings para numeros decimais
    df['Delivery_person_Ratings'] = df['Delivery_person_Ratings'].astype( float )

    # Conversao de texto para data
    df['Order_Date'] = pd.to_datetime( df['Order_Date'], format='%d-%m-%Y' )

    # Remove as linhas da culuna multiple_deliveries que tenham o 
    # conteudo igual a 'NaN '
    linhas_vazias = df['multiple_deliveries'] != 'NaN '
    df = df.loc[linhas_vazias, :]
    df['multiple_deliveries'] = df['multiple_deliveries'].astype( int )

    # Comando para remover o texto de números
    df = df.reset_index( drop=True )

    # Retirando os numeros da coluna Time_taken(min)
    df['Time_taken(min)'] = df['Time_taken(min)'].apply(lambda x: re.findall( r'\d+', x))

    # Retirando os espaços da coluna Festival
    df['Festival'] = df['Festival'].str.strip()
    df['City'] = df['City'].str.strip()
    df['Road_traffic_density'] = df['Road_traffic_density'].str.strip()

    # Remove os NAN da coluna City e Weatherconditions
    df = df.loc[df['City']!='NaN']
    df = df.loc[df['Weatherconditions'] != 'conditions NaN']

    # Remove os que forem np.na
    df = df.dropna()
    
    return df

def order_metric( df ):
    """ Esta função tem a responsabilidade de criar um gráfico de barras,
    que retorna a quantidade de pedidos por dia.
    
    Input: DataFrame
    Output: Gráfico de Barra
    """
    # Quantidade de pedidos por dia.
    df_grouped = df.loc[:, ['ID', 'Order_Date']].groupby( 'Order_Date').count().reset_index()
    df_grouped.columns = ['order_date', 'qtde_entregas']

    # Gráfico.
    fig = px.bar( df_grouped, x='order_date', y='qtde_entregas' )

    return fig

def traffic_order_share( df ):
    """ Esta função tem a responsabilidade de criar um gráfico de pizza,
    que retorna a quantidade de pedidos por dia.
    
    Input: DataFrame
    Output: Gráfico de Pizza
    """
    # Distribuição dos pedidos por tipo de tráfego. 
    df_grouped = df.loc[:, ['ID', 'Road_traffic_density']].groupby( 'Road_traffic_density' ).count().reset_index()
    df_grouped['perc_ID'] = 100 * ( df_grouped['ID'] / df_grouped['ID'].sum() )

    # Gráfico
    fig = px.pie( df_grouped, values='perc_ID', names='Road_traffic_density' )

    return fig

def order_by_week(df):
    """ Esta função tem a responsabilidade de criar um gráfico de linha,
    que retorna a quantidade de pedidos por semana.
    
    Input: DataFrame
    Output: Gráfico de linhas
    """
    # Quantidade de pedidos por Semana
    df['week_of_year'] = df['Order_Date'].dt.strftime( "%U" )
    df_aux = df.loc[:, ['ID', 'week_of_year']].groupby( 'week_of_year' ).count().reset_index()

    # gráfico
    fig = px.line( df_aux, x='week_of_year', y='ID' )

    return fig

def order_share_by_week(df):
    """ Esta função tem a responsabilidade de criar um gráfico de linha,
    que retorna a quantidade de pedidos dos entregadores por semana.
    
    Input: DataFrame
    Output: Gráfico de linhas
    """
    # Quantidade de pedidos por entregador por Semana
    df_aux1 = (df.loc[:, ['ID', 'week_of_year']]
               .groupby( 'week_of_year' )
               .count()
               .reset_index())
    df_aux2 = (df.loc[:, ['Delivery_person_ID', 'week_of_year']]
               .groupby( 'week_of_year')
               .nunique()
               .reset_index())
    df_aux = pd.merge( df_aux1, df_aux2, how='inner' )
    df_aux['order_by_delivery'] = df_aux['ID'] / df_aux['Delivery_person_ID']

    # gráfico
    fig = px.line( df_aux, x='week_of_year', y='order_by_delivery' )
    
    return fig

def country_maps(df):
    """ Esta função tem a responsabilidade de criar um mapa geográfico,
    que retorna a quantidade de pedidos dos entregadores por semana.
    
    Input: DataFrame
    Output: Mapa
    """
    data_plot = (df.loc[:, ['City', 'Road_traffic_density', 'Delivery_location_latitude', 'Delivery_location_longitude']]
                 .groupby( ['City', 'Road_traffic_density'])
                 .median().reset_index())                                                                                                                
    # Desenhar o mapa
    map = folium.Map( zoom_start=11 )
    for index, location_info in data_plot.iterrows():
        folium.Marker( [location_info['Delivery_location_latitude'],
                        location_info['Delivery_location_longitude']],
                        popup=location_info[['City', 'Road_traffic_density']] ).add_to( map )

    return map
# ----------------------- Inicio da Estrutura lógica do código-------------------#

# ==================================
# Configurações Página 
# ==================================
st.set_page_config(page_title='Visão Empresa', page_icon="🏢", layout='wide')

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

st.header('Marketplace - Visão Cliente')

image_path = 'logo.png'
image = Image.open(image_path)
st.sidebar.image(image, width=220)

st.sidebar.markdown('# Cury Company')
st.sidebar.markdown('## Fastest Delivery in Town')
st.sidebar.markdown("""---""")

st.sidebar.markdown('## Selecione uma data limite')

date_slider = st.sidebar.slider(
    'Até qual valor?',
    value=pd.to_datetime(2022,4,13),
    min_value=pd.to_datetime( 2022, 2, 11),
    max_value=pd.to_datetime( 2022, 4, 6),
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

tab1, tab2, tab3 = st.tabs( ['Visão Gerencial', 'Visão Tática', 'Visão Geografica'] )

with tab1:
    with st.container():
        st.markdown('# Orders by Day')
        fig = order_metric(df)
        st.plotly_chart( fig, use_container_with=True)
        
    with st.container():
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('## Traffic Order Share')
            fig = traffic_order_share(df)            
            st.plotly_chart( fig, use_container_width=True )

        with col2:
            st.markdown('## Traffic Order Share') 
            fig = traffic_order_share(df)            
            st.plotly_chart( fig, use_container_width=True )
    
with tab2:
    with st.container():
        st.markdown('## Order by Week ')
        fig = order_by_week(df)
        st.plotly_chart(fig, use_container_width=True)
        
    with st.container():
        st.markdown('### Order Share by Week')
        fig = order_share_by_week(df)
        st.plotly_chart(fig, use_container_width=True)
    
with tab3:
    st.markdown('# Country Maps')
    map = country_maps(df)
    folium_static(map, width= 1024, height=600)
    
