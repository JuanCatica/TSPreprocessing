from sklearn.base import BaseEstimator, TransformerMixin
from collections.abc import Iterable 
from itertools import chain

# --------------------------------------------
# CLASE PARA LAS COLUMNAS DEL DESFASE TEMPORAL 
#

class NLagDiff(BaseEstimator,TransformerMixin):
    """
    Esta clase permite aplicar dos la técnica para la extracción de caracteristicas, las cuales son:
    
    Técnica N-Lag:
        Esta permite asignar el vector de la vaiable objetivo con cieta cantidad de desplazamientos 
        al conjunto de variables (Matriz X). 
        
        Es importante mencionar que estos desplazamientos representan los valores historicos de la 
        variable objetivo, de esta forma durante la aplicacón de esta técnica no se incurre en "Data Spooning".
        Ya que los datos de ventanas temprales anteriores pueden se empleados como variabes de entrada en 
        el modelo, y pueden ser reutilizados en producción de manera iterativa y recusiva.
    
    Técnica Diff (T1 - T2):
        Esta técnica permite obtener la diferencia entre los valores de las dos últimas ventanas de tiempo.
        Al igual que la ténica N-Lag no incurre en  "Data Spooning" y es análogo a la aceleración en terminos
        de los cambios en la vriable objetivo.
    
    Este metodo permite crear una instacia de la clase NLagDiff
    
    Parametros:
    -----------

        columns: type list, Default list().
            Se refiere a las columnas de las variables objetivoos, es decir, aquellas a las 
            que se le aplicaran la técnica N-Lag y Diff (Esta se puede incluir o noor medio 
            el paramtro "diff").
                    
        lags: type int or list, Default 1.
            Este parametro hace referencia los lags o desplazamientos temporales  que serán ejecutados 
            bajo las técnicas N-Lag y Diff (Esta ultima se puede incluir o no por medio del paramtro "diff").
            
            Si el valor es positivo se desarrolla un desplazamiento hacia adelante, es decir, en terminos de
            un dataframe las columnas definidas en 'columns' serán desplazadas hacia abajo. Esto significa 
            que los registros "antiguos" quedan alineados con registros "contemporaneos" con la finalidad
            de poder extraer relaciones temporales por medio de los estimadores que se implementen.
            
            Si por el contrario el valor es negativo los desplazamientos temporales se ejecutarán hacia atras,
            y en terminos de un dataframe las columnas serán desplazadas hacia arriba. Esto significa que se 
            puede aliniear registros "futuros" con con registros "contemporaneos"  con la finalidad de relacionar
            futuros valores sobre algunas variables con lo que ha tenido lugar en el pasado por medio de los 
            estimadoes que se lleguen a implementar.

            Importante: El termino contemporaneo es subjetivo y hace referencia a un registro en particular
                        que se puede tomar como el registro actual por efectos de entendimiento.
                
        col_group: Default None.
            Este parametro se empla para desarrollar agrupaciones previas a la aplicación de las
            dos técnicas. Si el valor se encuentrá en "None" no se desarrolla ningún tipo de agrupación.

            Este parametro es util cuando el dataset al que se le aplicarán las técnicas contine datos temporales
            de distintos grupos de entidades. Un ejemplo es un dataset temporal de ventas de difetentes artículos
            de una tienda y se desea realizar los desfases para cada uno de la serie de cada artículo.
        
        dropna: type bool, default False.
            Detemina si se elimina o no aquellos registros con registros faltantes una vez se ha termnado
            la ejecicón de las técnicas para la extraciónd e caracteristicas, si el valor es "True" se 
            desarrolla la eliminación (inplace = True).
            
            Importante: Puede elimnar regristos de manera indeseada.
            
        diff: type bool, default True.
            Determina si se implementara la tecnica "Diff".

        copy: type bool, default True.
            Determina si se desarrollará una copia del dataframe que se desea transformar.

    Ejemplo:
    --------

        Se selecciona unicamente la columna 'X1' y el valor de desplazamiento en lags es igual a 1.
        Por tanto el desplazamiento es hacia adelante (Trae valores del pasado).

        TABLA_LAG_DIFF = NLagDiff(columns = ["X1"],lags = 1).fit_transform(TABLA);

        ______TABLA______             ________________TABLA_LAG_DIFF___________________
        |index| X1 |  Y |             |index|  Y | X1 | X1_lag_1back |  X1_diff_1back |
        |  0  |  1 |  1 |             |  0  |  1 |  1 |     NaN      |       NaN      |
        |  1  |  2 |  2 |             |  1  |  2 |  2 |       1      |       NaN      |
        |  2  |  3 |  3 |             |  2  |  3 | -3 |       2      |         1      |
        |  3  |  4 |  4 |             |  3  |  4 |  4 |      -3      |        -5      |
        |  4  |  5 |  5 | Lag=[1] --> |  4  |  5 |  8 |       4      |         7      |
        |  5  |  6 |  6 |             |  5  |  6 |  6 |       8      |         4      |
        |  6  |  7 |  7 |             |  6  |  7 |  7 |       6      |        -2      |
        |  7  |  8 |  8 |             |  7  |  8 |  8 |       7      |         1      |
        |  8  |  9 |  9 |             |  8  |  9 |  9 |       8      |         1      |
        |  9  | 10 | 10 |             |  9  | 10 | 10 |       9      |         1      |
        | 10  | 11 | 11 |             | 10  | 11 | 11 |      10      |         1      |
        |_____|____|____|             |_____|____|____|______________|________________|

    Desarrollador:
        Ing. Juan Camilo Cática Buendía
    """
    def __init__(self, columns = list(), lags = 1, col_group = None, dropna = False, diff = True, copy = True):

        self._columns = columns
        self._col_group = col_group
        self._dropna = dropna
        self._diff = diff
        self._copy = copy
        
        if not isinstance(lags, Iterable):
            self._lags = [lags]
    
    def fit(self, X, y=None):
        """
        Este método se hereda de la clase BaseEstimator y permite suministrar los datos de entrada.
        
        Parametros
        ----------
        X: type pandas.Dtaframe
            Se espera que X sean una instancia de la clase Dataframe puesto que es posible desarrollar
            agrupaciones sobre esta (self._X) en el método "self.transform()"
        """
        if self._copy:
            self._X = X.copy()
        else:
            self._X = X
        return self
    
    def transform(self, X):
        """
        En este metodo se desarrolla la extracción de caracteristicas bajo las técnicas N-Lag y Diff.
        
        Inicialmente este método toma cada uno de los valores de lags o diferencias en las ventanas 
        de tiempo que se desean. A partir de esto genera los nombres de las nuevas variables a partir de las
        variables objetivos, asignandoles prefijos. Posteriormente desarrolla la extracción de las caracteristcas 
        bajo las técnicas consultando si de debe realizar agrupaciones en la data.
        
         Parametros
        ----------
        X: type pandas.Dtaframe
            Se espera que X sean una instancia de la clase Dataframe puesto que es posible desarrollar
            agrupaciones.
        """
        for i in self._lags:
            sufix = f"{i}back" if i > 0 else f"{-i}forward"
            
            new_lag_columns = list(map(lambda x: f"{x}_lag_{sufix}", self._columns))
            new_diff_columns = list(map(lambda x: f"{x}_diff_{sufix}", self._columns))
            if self._col_group:
                self._X[new_lag_columns] = self._X.groupby(self._col_group)[self._columns].shift(i)
                if self._diff:
                    self._X[new_diff_columns] = self._X[new_lag_columns] - self._X.groupby(self._col_group)[new_lag_columns].shift(i)
            else:            
                self._X[new_lag_columns] = self._X[self._columns].shift(i)
                if self._diff:
                    self._X[new_diff_columns] = self._X[new_lag_columns] - self._X[new_lag_columns].shift(i)
                
        if self._dropna:
            self._X.dropna(inplace = True)
            self._X = self._X.reset_index(drop=True)
        
        return self._X   


# --------------------------------------------------------------------------------
# CLASE PARA GENERAR LOS FOLDS ADECUADOS EN UN CONJUNTO DE DATOS DE SERIE TEMPORAL
#

class TimeSeriesKFold:
    """
    Esta clase se emplea para crear un generado que entrega el conjunto de train y test bajo el 
    método de Encadenamiento hacia adelante (Forward-Chaning) o Vetana Deslizante (Sliding window)
    
    Es importante mencionar que el  dataframe que se procesa por medio de los objetos de esta clase deben 
    contener los registros ordenados por fechas de manera creciente al igual que el indice de dicho dataframe,
    el cual debe estar ordenado de manera incremental desde el valor cero hasta el número de registos - 1.
    
    Iniialización de la clase "ForwardChaningKFold".
    
    Parametros:
    -----------
        
        y: str 
            Nombre de la columna que se desea emplear como variable a predecir.
            
        init: int
            Valor de inicio de la ventana temporal
        
        final: int
            Valor de finalización de el conjunto de iteraciones.
            
        win_size: int
            Tamaño de la ventana. Para el método 'forward_chaining' esta este valor aumenta iterativamente, 
            mientras que en 'sliding_window' el valor que se especifia permanece fijo.
            
        method: str ('forward_chaining' o 'sliding_window')
            Método de validación para conjunto de datos temporales.

    Ejemplos:
    ---------

        Metodo 'forward_chaining':

            Bajo este metodo el tamaño de la venta de entrenamiento 'win_size' no es fijo y se incrementa
            de manera iteratva tras la generación de cada uno de los conjuntos tran-test hasta lograr
            que el valor de  'init + win_size' sea igual a 'final - 1'.
            
            Ejemplo:
            init     = 1
            win_size = 5
            final    = 10
                     ______iter 1_____            ______iter 2_____              ______iter 3_____
                     |index| X1 |  Y |            |index| X1 |  Y |              |index| X1 |  Y |
                     |  0  |  1 |  1 |            |  0  |  1 |  1 |              |  0  |  1 |  1 |  
            init *-->|  1  | 42 | 42 |   init *-->|  1  | 42 | 42 |     init *-->|  1  | 42 | 42 |
                 |   |  2  | 15 | 15 |        |   |  2  | 15 | 15 |          |   |  2  | 15 | 15 |
                 |   |  3  | 48 | 48 |        |   |  3  | 48 | 48 |          |   |  3  | 48 | 48 |
            win_size |  4  | 95 | 95 |        |   |  4  | 95 | 95 |          |   |  4  | 95 | 95 |
                 |   |  5  |  8 |  8 |   win_size |  5  |  8 |  8 |          |   |  5  |  8 |  8 |
                 |   |  6  | 12 | 12 |        |   |  6  | 12 | 12 |     win_size |  6  | 12 | 12 |
                 *-->|  7  | 42 | 42 |        |   |  7  | 42 | 42 |          |   |  7  | 42 | 42 |
            test---->|  8  | 15 | 15 |        *-->|  8  | 15 | 15 |          |   |  8  | 15 | 15 |
                     |  9  | 48 | 48 |   test---->|  9  | 48 | 48 |          *-->|  9  | 48 | 48 |
            final--->| 10  | 95 | 95 |   final--->| 10  | 95 | 95 | final & test>| 10  | 95 | 95 |
                     |_____|____|____|            |_____|____|____|              |_____|____|____|

        Metodo 'sliding_window':

            Bajo este metodo el tamaño de la venta de entrenamiento 'win_size' permanece siempre fijo y
            el valor de 'init' aumenta progresivamente, de esta froma se logra obtener siempre conjuntos
            del mismo tamaño pero en posiciones diferentes, que simulan el avance en el tiempo. El poceso 
            iterativo se desarrolla hasta lograr que el valor de 'init + win_size' sea igual a 'final - 1'.
            
            Ejemplo:
              init     = 1
              win_size = 5
              final    = 10
                     ______iter 1_____            ______iter 2_____              ______iter 3_____
                     |index| X1 |  Y |            |index| X1 |  Y |              |index| X1 |  Y |
                     |  0  |  1 |  1 |            |  0  |  1 |  1 |              |  0  |  1 |  1 |  
            init *-->|  1  | 42 | 42 |            |  1  | 42 | 42 |              |  1  | 42 | 42 |
                 |   |  2  | 15 | 15 |   init *-->|  2  | 15 | 15 |              |  2  | 15 | 15 |
                 |   |  3  | 48 | 48 |        |   |  3  | 48 | 48 |     init *-->|  3  | 48 | 48 |
            win_size |  4  | 95 | 95 |        |   |  4  | 95 | 95 |          |   |  4  | 95 | 95 |
                 |   |  5  |  8 |  8 |   win_size |  5  |  8 |  8 |          |   |  5  |  8 |  8 |
                 |   |  6  | 12 | 12 |        |   |  6  | 12 | 12 |     win_size |  6  | 12 | 12 |
                 *-->|  7  | 42 | 42 |        |   |  7  | 42 | 42 |          |   |  7  | 42 | 42 |
            test---->|  8  | 15 | 15 |        *-->|  8  | 15 | 15 |          |   |  8  | 15 | 15 |
                     |  9  | 48 | 48 |   test---->|  9  | 48 | 48 |          *-->|  9  | 48 | 48 |
            final--->| 10  | 95 | 95 |   final--->| 10  | 95 | 95 | final & test>| 10  | 95 | 95 |
                     |_____|____|____|            |_____|____|____|              |_____|____|____|

    Desarrollador:
        Ing. Juan Camilo Cática Buendía
    """
    
    def __init__(self, y, init, final, win_size, method = "forward_chaining"):

        self._y = y
        self._init = init
        self._final = final
        self._win_size = win_size
        self._method = method.lower()
        
        if ((y==None) | (init==None) | (final==None) | (win_size==None)):
            raise TypeError("Entradas Incompletas")
                
        if self._method not in ["sliding_window","forward_chaining"]:
            raise TypeError("Metodo incorrecto. Opciones: 'sliding_window' o 'forward_chaining' ")
                
    def _train_test_split_forward_chaning(self,X):
        """
        Este metodo se ejecuta cuando se pasa 'forward_chaining' a través del parametro 'method' del metodo 
        constructor '__init__()'.

        Ver esquema de funcionamiento de este método en el docstring de definición de esta clase.
        """
        for i in range(self._init+self._win_size, self._final):
            train = X.iloc[self._init : i]
            val   = X.iloc[i:i+1]
            X_train, X_test = train.drop([self._y], axis=1), val.drop([self._y], axis=1)
            y_train, y_test = train[self._y].values, val[self._y].values
            yield X_train, X_test, y_train, y_test
            
    def _train_test_split_sliding_window(self,X):
        """ 
        Este metodo se ejecuta cuando se pasa 'sliding_window' a través del parametro 'method' del metodo 
        constructor '__init__()'.

        Ver esquema de funcionamiento de este método en el docstring de definición de esta clase.
        """            
        recorrido = self._final - (self._init + self._win_size)
        for i in range(recorrido):
            init_train = self._init + i
            final_train = init_train + self._win_size
            
            train = X.iloc[init_train:final_train]
            val   = X.iloc[final_train:final_train+1]
            X_train, X_test = train.drop([self._y], axis=1), val.drop([self._y], axis=1)
            y_train, y_test = train[self._y].values, val[self._y].values
            yield X_train, X_test, y_train, y_test
    
    def split(self,X):
        """
        Metodo para ejecutar el tipo de validación cuzada especificada durante la instanciación de la clase.
        
        En este metodo se desarrolla el control de inconsistencias según los valores de 'init', 'final' 
        y 'win_size' con respecto al parametro de entrada 'X'
        
        Parametros:
        
        X: DataFrame
            Conjunto de datos organizados temporalmente de manera creciente y con valores de los indices
            igualmente organizados desde el valor 0 hasta el valor la longitud del DataFrame - 1.
        """
        x_size = len(X)
        self._final = x_size if x_size < (self._final-1) else self._final
        recorrido = self._final - (self._init + self._win_size)
        
        if not x_size:
            raise ValueError("Se debe ingresar al menos un elemento")     
        
        if recorrido < 0 or self._win_size > x_size:
            raise ValueError("Los prametros 'init', 'final' o 'win_size' no son apropiados")
        
        if self._method == "forward_chaining":
            return chain(self._train_test_split_forward_chaning(X))
        elif self._method == "sliding_window":
            return chain(self._train_test_split_sliding_window(X))
        else:
            raise ValueError("Metodo no especificado")     