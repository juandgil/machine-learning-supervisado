import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
from fuzzywuzzy import process  # Para comparar cadenas

# Cargar los datos desde un archivo CSV
data = pd.read_csv('datos_transporte_supervisado.csv')

# Limpiar los nombres de las columnas
data.columns = data.columns.str.strip()  # Eliminar espacios en blanco en los nombres de las columnas

# Preprocesamiento de datos
label_encoder_trafico = LabelEncoder()  # Inicializar el codificador para la variable 'tráfico'
label_encoder_clima = LabelEncoder()  # Inicializar el codificador para la variable 'clima'

# Ajustar el LabelEncoder con las categorías existentes
data['tráfico'] = label_encoder_trafico.fit_transform(data['Tráfico'])  # Codificar la columna 'Tráfico'
data['clima'] = label_encoder_clima.fit_transform(data['Clima'])  # Codificar la columna 'Clima'

# Definir las variables independientes y dependientes
X = data[['Distancia (km)', 'tráfico', 'clima']]  # Variables independientes (features)
y_llegada = data['Tiempo de llegada (min)']  # Variable dependiente para llegada
y_regreso = data['Tiempo de regreso (min)']  # Variable dependiente para regreso

# Dividir los datos en conjunto de entrenamiento y prueba
X_train, X_test, y_train_llegada, y_test_llegada = train_test_split(X, y_llegada, test_size=0.2, random_state=42)
X_train, X_test, y_train_regreso, y_test_regreso = train_test_split(X, y_regreso, test_size=0.2, random_state=42)

# Crear el modelo para llegada
modelo_llegada = RandomForestRegressor()  # Inicializar el modelo de regresión de bosque aleatorio
modelo_llegada.fit(X_train, y_train_llegada)  # Entrenar el modelo con los datos de entrenamiento

# Hacer predicciones para llegada
predicciones_llegada = modelo_llegada.predict(X_test)  # Predecir los tiempos de llegada

# Evaluar el modelo para llegada
error_llegada = mean_squared_error(y_test_llegada, predicciones_llegada)  # Calcular el error cuadrático medio
print(f'Error cuadrático medio (Llegada): {error_llegada}')  # Imprimir el error

# Crear el modelo para regreso
modelo_regreso = RandomForestRegressor()  # Inicializar otro modelo de regresión de bosque aleatorio
modelo_regreso.fit(X_train, y_train_regreso)  # Entrenar el modelo con los datos de regreso

# Hacer predicciones para regreso
predicciones_regreso = modelo_regreso.predict(X_test)  # Predecir los tiempos de regreso

# Evaluar el modelo para regreso
error_regreso = mean_squared_error(y_test_regreso, predicciones_regreso)  # Calcular el error cuadrático medio
print(f'Error cuadrático medio (Regreso): {error_regreso}')  # Imprimir el error

# Función para sugerir el municipio más cercano
def sugerir_municipio(municipio):
    municipios = data['Municipio'].tolist()  # Obtener la lista de municipios
    mejor_coincidencia = process.extractOne(municipio, municipios)  # Encontrar la mejor coincidencia
    return mejor_coincidencia[0] if mejor_coincidencia[1] >= 70 else None  # Retornar la coincidencia si es suficientemente alta

# Función para sugerir tráfico y clima más cercanos
def sugerir_categoria(categoria, tipo):
    # Obtener las categorías únicas del DataFrame según el tipo
    if tipo == 'tráfico':
        categorias = data['Tráfico'].unique()  # Extraer categorías de la columna 'Tráfico'
    else:
        categorias = data['Clima'].unique()  # Extraer categorías de la columna 'Clima'
    
    mejor_coincidencia = process.extractOne(categoria, categorias)  # Encontrar la mejor coincidencia
    return mejor_coincidencia[0] if mejor_coincidencia[1] >= 70 else None  # Retornar la coincidencia si es suficientemente alta

# Función para hacer preguntas al modelo
def predecir_tiempos(municipio, trafico, clima):
    # Obtener la distancia del municipio seleccionado
    distancia = data.loc[data['Municipio'].str.lower() == municipio.lower(), 'Distancia (km)'].values
    if len(distancia) == 0:
        return None, None  # Retornar None si el municipio no es válido
    distancia = distancia[0]
    
    # Codificar las variables categóricas
    try:
        trafico_encoded = label_encoder_trafico.transform([trafico])[0]  # Codificar tráfico
        clima_encoded = label_encoder_clima.transform([clima])[0]  # Codificar clima
    except ValueError:
        print("Error: Asegúrate de que las entradas de tráfico y clima sean válidas.")
        return None, None
    
    # Crear un DataFrame para la entrada
    entrada = pd.DataFrame([[distancia, trafico_encoded, clima_encoded]], columns=['Distancia (km)', 'tráfico', 'clima'])
    
    # Hacer la predicción para llegada
    tiempo_llegada_predicho = modelo_llegada.predict(entrada)[0]  # Predecir tiempo de llegada
    
    # Hacer la predicción para regreso
    tiempo_regreso_predicho = modelo_regreso.predict(entrada)[0]  # Predecir tiempo de regreso
    
    return tiempo_llegada_predicho, tiempo_regreso_predicho  # Retornar las predicciones

# uso de la función
if __name__ == "__main__":
    # Mostrar los municipios disponibles
    print("Municipios disponibles:")
    print(data['Municipio'].unique())
    
    municipio = input("Ingrese el municipio al que desea ir desde Medellín: ")
    if municipio not in data['Municipio'].values:  # Comparar directamente sin normalización
        sugerencia = sugerir_municipio(municipio)
        if sugerencia:
            print(f"Asumo que te refieres a '{sugerencia}'.")
            municipio = sugerencia  # Usar la sugerencia como municipio válido
    
    trafico = input("Ingrese el tráfico (Bajo, Moderado, Alto): ")
    if trafico not in ['Bajo', 'Moderado', 'Alto']:
        sugerencia_trafico = sugerir_categoria(trafico, 'tráfico')
        if sugerencia_trafico:
            print(f"Asumo que te refieres a '{sugerencia_trafico}' para el tráfico.")
            trafico = sugerencia_trafico  # Usar la sugerencia como tráfico válido
    
    clima = input("Ingrese el clima (Soleado, Nublado, Lluvia): ")
    if clima not in ['Soleado', 'Nublado', 'Lluvia']:
        sugerencia_clima = sugerir_categoria(clima, 'clima')
        if sugerencia_clima:
            print(f"Asumo que te refieres a '{sugerencia_clima}' para el clima.")
            clima = sugerencia_clima  # Usar la sugerencia como clima válido

    # print(trafico)
    # print(clima)
    # print(municipio)
    
    # Validar nuevamente las entradas antes de predecir
    if trafico in ['Bajo', 'Moderado', 'Alto'] and clima in ['Soleado', 'Nublado', 'Lluvia']:
        tiempo_llegada, tiempo_regreso = predecir_tiempos(municipio, trafico, clima)
        if tiempo_llegada is not None and tiempo_regreso is not None:
            print(f"El tiempo de llegada estimado es: {tiempo_llegada:.2f} minutos")
            print(f"El tiempo de regreso estimado es: {tiempo_regreso:.2f} minutos")
    else:
        print("Error: Las entradas de tráfico y clima no son válidas.")