import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import xgboost as xgb
from catboost import CatBoostRegressor
import warnings
warnings.filterwarnings('ignore')

class CognitiveBehaviorAnalyzer:
    def __init__(self, data_path):
        """Inicializar analizador con datos del TFM"""
        self.df = pd.read_csv(data_path)
        self.models = {}
        self.clusters_data = {}
        self.scaler = StandardScaler()
        
    def explore_data(self):
        """Análisis exploratorio de datos"""
        print("=== ANÁLISIS EXPLORATORIO ===")
        print(f"Dimensiones del dataset: {self.df.shape}")
        print(f"Columnas disponibles: {list(self.df.columns)}")
        print("\nPrimeras 5 filas:")
        print(self.df.head())
        
        # Estadísticas descriptivas
        print("\n=== ESTADÍSTICAS DESCRIPTIVAS ===")
        numeric_cols = ['Tiempo_respuesta', 'Distancia.t', 'Error.t']
        print(self.df[numeric_cols].describe())
        
        # Distribución por género y tipo de prueba
        print("\n=== DISTRIBUCIÓN CATEGÓRICA ===")
        print("Por género:")
        print(self.df['Genero'].value_counts())
        print("\nPor tipo de prueba:")
        print(self.df['CrucesColisiones'].value_counts())
        
        return self.df.info()
    
    def prepare_data_for_prediction(self, target_variable='Distancia.t'):
        """Preparar datos para modelos predictivos"""
        print(f"\n=== PREPARACIÓN DE DATOS PARA PREDICCIÓN DE {target_variable} ===")
        
        # Variables predictoras
        feature_cols = ['Genero', 'CrucesColisiones', 'Trial', 'Tiempo_respuesta']
        
        # Crear dataset sin valores nulos
        clean_data = self.df[feature_cols + [target_variable]].dropna()
        
        # Codificar variables categóricas
        X = pd.get_dummies(clean_data[feature_cols], drop_first=True)
        y = clean_data[target_variable]
        
        # División train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=clean_data['Genero']
        )
        
        print(f"Tamaño entrenamiento: {X_train.shape}")
        print(f"Tamaño test: {X_test.shape}")
        print(f"Features utilizadas: {list(X.columns)}")
        
        return X_train, X_test, y_train, y_test, X.columns
    
    def compare_predictive_models(self, target='Distancia.t'):
        """Comparar diferentes modelos predictivos"""
        print(f"\n=== COMPARACIÓN DE MODELOS PREDICTIVOS ===")
        
        X_train, X_test, y_train, y_test, feature_names = self.prepare_data_for_prediction(target)
        
        # Definir modelos
        models = {
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42, max_depth=8),
            'XGBoost': xgb.XGBRegressor(n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42),
            'CatBoost': CatBoostRegressor(iterations=100, depth=4, learning_rate=0.1, random_state=42, verbose=False)
        }
        
        results = {}
        
        for name, model in models.items():
            print(f"\nEntrenando {name}...")
            
            # Entrenar modelo
            model.fit(X_train, y_train)
            
            # Predicciones
            y_pred_train = model.predict(X_train)
            y_pred_test = model.predict(X_test)
            
            # Métricas
            train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
            test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
            test_r2 = r2_score(y_test, y_pred_test)
            test_mae = mean_absolute_error(y_test, y_pred_test)
            
            # Cross-validation
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
            cv_rmse = np.sqrt(-cv_scores.mean())
            
            results[name] = {
                'model': model,
                'train_rmse': train_rmse,
                'test_rmse': test_rmse,
                'test_r2': test_r2,
                'test_mae': test_mae,
                'cv_rmse': cv_rmse,
                'predictions': y_pred_test
            }
            
            print(f"RMSE Train: {train_rmse:.4f}")
            print(f"RMSE Test: {test_rmse:.4f}")
            print(f"R² Test: {test_r2:.4f}")
            print(f"MAE Test: {test_mae:.4f}")
            print(f"CV RMSE: {cv_rmse:.4f}")
        
        # Guardar mejores modelos
        self.models[target] = results
        self.test_data = (X_test, y_test)
        
        return results
    
    def visualize_model_performance(self, target='Distancia.t'):
        """Visualizar rendimiento de modelos"""
        if target not in self.models:
            print("Primero ejecuta compare_predictive_models()")
            return
        
        results = self.models[target]
        X_test, y_test = self.test_data
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Comparación de métricas
        model_names = list(results.keys())
        rmse_scores = [results[m]['test_rmse'] for m in model_names]
        r2_scores = [results[m]['test_r2'] for m in model_names]
        
        axes[0,0].bar(model_names, rmse_scores, color=['skyblue', 'lightgreen', 'salmon'])
        axes[0,0].set_title('RMSE por Modelo')
        axes[0,0].set_ylabel('RMSE')
        axes[0,0].tick_params(axis='x', rotation=45)
        
        axes[0,1].bar(model_names, r2_scores, color=['skyblue', 'lightgreen', 'salmon'])
        axes[0,1].set_title('R² por Modelo')
        axes[0,1].set_ylabel('R²')
        axes[0,1].tick_params(axis='x', rotation=45)
        
        # 2. Predicciones vs Realidad (mejor modelo)
        best_model_name = min(results.keys(), key=lambda x: results[x]['test_rmse'])
        best_predictions = results[best_model_name]['predictions']
        
        axes[1,0].scatter(y_test, best_predictions, alpha=0.6, color='blue')
        axes[1,0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        axes[1,0].set_xlabel('Valores Reales')
        axes[1,0].set_ylabel('Predicciones')
        axes[1,0].set_title(f'Predicciones vs Realidad - {best_model_name}')
        axes[1,0].grid(True, alpha=0.3)
        
        # 3. Importancia de variables (XGBoost)
        if 'XGBoost' in results:
            xgb_model = results['XGBoost']['model']
            importances = xgb_model.feature_importances_
            feature_names = X_test.columns
            
            # Ordenar por importancia
            indices = np.argsort(importances)[::-1]
            
            axes[1,1].bar(range(len(importances)), importances[indices], color='green', alpha=0.7)
            axes[1,1].set_title('Importancia de Variables (XGBoost)')
            axes[1,1].set_xlabel('Variables')
            axes[1,1].set_ylabel('Importancia')
            axes[1,1].set_xticks(range(len(importances)))
            axes[1,1].set_xticklabels([feature_names[i] for i in indices], rotation=45)
        
        plt.tight_layout()
        plt.show()
        
        print(f"\n=== MEJOR MODELO ===")
        print(f"Modelo: {best_model_name}")
        print(f"RMSE: {results[best_model_name]['test_rmse']:.4f}")
        print(f"R²: {results[best_model_name]['test_r2']:.4f}")
    
    def cognitive_clustering(self, n_clusters_range=(2, 8)):
        """Análisis de clustering para perfiles cognitivos"""
        print("\n=== ANÁLISIS DE CLUSTERING COGNITIVO ===")
        
        # Variables para clustering
        cluster_vars = ['Tiempo_respuesta', 'Distancia.t', 'Error.t']
        cluster_data = self.df[cluster_vars].dropna()
        
        # Estandarizar datos
        X_scaled = self.scaler.fit_transform(cluster_data)
        
        # Encontrar número óptimo de clusters
        silhouette_scores = []
        K_range = range(n_clusters_range[0], n_clusters_range[1] + 1)
        
        for k in K_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(X_scaled)
            silhouette_avg = silhouette_score(X_scaled, cluster_labels)
            silhouette_scores.append(silhouette_avg)
        
        # Visualizar curva de silhouette
        plt.figure(figsize=(10, 6))
        plt.plot(K_range, silhouette_scores, marker='o', linewidth=2, markersize=8)
        plt.title('Silhouette Score vs Número de Clusters')
        plt.xlabel('Número de Clusters (K)')
        plt.ylabel('Silhouette Score')
        plt.grid(True, alpha=0.3)
        plt.show()
        
        # Seleccionar mejor K
        optimal_k = K_range[np.argmax(silhouette_scores)]
        print(f"Número óptimo de clusters: {optimal_k}")
        print(f"Silhouette Score: {max(silhouette_scores):.3f}")
        
        # Clustering final
        kmeans_final = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
        cluster_labels = kmeans_final.fit_predict(X_scaled)
        
        # Añadir clusters al dataframe
        cluster_df = cluster_data.copy()
        cluster_df['cluster'] = cluster_labels
        
        # Análisis de perfiles
        print("\n=== PERFILES COGNITIVOS ===")
        cluster_profiles = cluster_df.groupby('cluster')[cluster_vars].agg(['mean', 'std']).round(3)
        print(cluster_profiles)
        
        # Guardar resultados
        self.clusters_data = {
            'data': cluster_df,
            'scaled_data': X_scaled,
            'labels': cluster_labels,
            'model': kmeans_final,
            'optimal_k': optimal_k,
            'profiles': cluster_profiles
        }
        
        return cluster_df, cluster_labels
    
    def visualize_cognitive_clusters(self):
        """Visualizar clusters cognitivos"""
        if not self.clusters_data:
            print("Primero ejecuta cognitive_clustering()")
            return
        
        cluster_df = self.clusters_data['data']
        X_scaled = self.clusters_data['scaled_data']
        labels = self.clusters_data['labels']
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. PCA 2D
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)
        
        scatter = axes[0,0].scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='tab10', alpha=0.7)
        axes[0,0].set_title('Clusters Cognitivos (PCA 2D)')
        axes[0,0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} varianza)')
        axes[0,0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} varianza)')
        axes[0,0].grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=axes[0,0])
        
        # 2. Distribución por género
        df_with_clusters = self.df.copy()
        df_with_clusters = df_with_clusters.loc[cluster_df.index]
        df_with_clusters['cluster'] = labels
        
        gender_cluster = pd.crosstab(df_with_clusters['cluster'], df_with_clusters['Genero'], normalize='index')
        gender_cluster.plot(kind='bar', ax=axes[0,1], stacked=True)
        axes[0,1].set_title('Distribución de Género por Cluster')
        axes[0,1].set_ylabel('Proporción')
        axes[0,1].tick_params(axis='x', rotation=0)
        axes[0,1].legend(title='Género')
        
        # 3. Perfiles de rendimiento
        cluster_means = cluster_df.groupby('cluster')[['Tiempo_respuesta', 'Distancia.t', 'Error.t']].mean()
        cluster_means.plot(kind='bar', ax=axes[1,0])
        axes[1,0].set_title('Perfiles de Rendimiento por Cluster')
        axes[1,0].set_ylabel('Valor Promedio')
        axes[1,0].tick_params(axis='x', rotation=0)
        axes[1,0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # 4. Heatmap de correlaciones por cluster
        correlations = []
        for cluster_id in sorted(cluster_df['cluster'].unique()):
            cluster_subset = cluster_df[cluster_df['cluster'] == cluster_id][['Tiempo_respuesta', 'Distancia.t', 'Error.t']]
            corr_matrix = cluster_subset.corr()
            correlations.append(corr_matrix.values[np.triu_indices(3, k=1)])
        
        # Matriz de correlaciones promedio
        avg_corr = np.mean(correlations, axis=0)
        corr_labels = ['Tiempo-Distancia', 'Tiempo-Error', 'Distancia-Error']
        
        im = axes[1,1].imshow([avg_corr], cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
        axes[1,1].set_title('Correlaciones Promedio Entre Variables')
        axes[1,1].set_xticks(range(len(corr_labels)))
        axes[1,1].set_xticklabels(corr_labels, rotation=45)
        axes[1,1].set_yticks([0])
        axes[1,1].set_yticklabels(['Promedio'])
        plt.colorbar(im, ax=axes[1,1])
        
        plt.tight_layout()
        plt.show()
    
    def interpret_cognitive_profiles(self):
        """Interpretar perfiles cognitivos encontrados"""
        if not self.clusters_data:
            print("Primero ejecuta cognitive_clustering()")
            return
        
        cluster_df = self.clusters_data['data']
        profiles = self.clusters_data['profiles']
        
        print("\n=== INTERPRETACIÓN DE PERFILES COGNITIVOS ===")
        
        for cluster_id in sorted(cluster_df['cluster'].unique()):
            cluster_data = cluster_df[cluster_df['cluster'] == cluster_id]
            n_participants = len(cluster_data)
            
            # Estadísticas del cluster
            tiempo_mean = profiles.loc[cluster_id, ('Tiempo_respuesta', 'mean')]
            distancia_mean = profiles.loc[cluster_id, ('Distancia.t', 'mean')]
            error_mean = profiles.loc[cluster_id, ('Error.t', 'mean')]
            
            print(f"\n--- CLUSTER {cluster_id} ({n_participants} participantes) ---")
            print(f"Tiempo de respuesta promedio: {tiempo_mean:.2f}s")
            print(f"Distancia promedio: {distancia_mean:.2f}")
            print(f"Error promedio: {error_mean:.2f}")
            
            # Caracterización del perfil
            if tiempo_mean < cluster_df['Tiempo_respuesta'].mean():
                speed_profile = "RÁPIDOS"
            else:
                speed_profile = "LENTOS"
            
            if distancia_mean < cluster_df['Distancia.t'].mean():
                accuracy_profile = "PRECISOS"
            else:
                accuracy_profile = "IMPRECISOS"
            
            print(f"Perfil: {speed_profile} y {accuracy_profile}")
            
            # Estrategia cognitiva inferida
            if speed_profile == "RÁPIDOS" and accuracy_profile == "PRECISOS":
                strategy = "Procesamiento eficiente - Alta competencia cognitiva"
            elif speed_profile == "RÁPIDOS" and accuracy_profile == "IMPRECISOS":
                strategy = "Estrategia impulsiva - Velocidad sobre precisión"
            elif speed_profile == "LENTOS" and accuracy_profile == "PRECISOS":
                strategy = "Estrategia deliberativa - Precisión sobre velocidad"
            else:
                strategy = "Procesamiento ineficiente - Posibles dificultades"
            
            print(f"Estrategia cognitiva: {strategy}")
    
    def advanced_feature_engineering(self):
        """Crear características avanzadas para mejores predicciones"""
        print("\n=== INGENIERÍA DE CARACTERÍSTICAS AVANZADAS ===")
        
        # Features basadas en tendencias temporales
        self.df = self.df.sort_values(['ID', 'Trial'])
        
        # Calcular tendencias por participante
        advanced_features = []
        
        for participant_id in self.df['ID'].unique():
            participant_data = self.df[self.df['ID'] == participant_id].copy()
            
            if len(participant_data) < 3:  # Necesitamos al menos 3 trials
                continue
            
            # Tendencias temporales
            participant_data['mejora_distancia'] = -participant_data['Distancia.t'].diff()  # Negativo porque menor distancia es mejor
            participant_data['mejora_error'] = -participant_data['Error.t'].diff()
            participant_data['cambio_tiempo'] = participant_data['Tiempo_respuesta'].diff()
            
            # Estadísticas de ventana móvil
            participant_data['distancia_avg_3'] = participant_data['Distancia.t'].rolling(window=3, min_periods=1).mean()
            participant_data['tiempo_avg_3'] = participant_data['Tiempo_respuesta'].rolling(window=3, min_periods=1).mean()
            
            # Variabilidad
            participant_data['distancia_std_3'] = participant_data['Distancia.t'].rolling(window=3, min_periods=1).std()
            participant_data['tiempo_std_3'] = participant_data['Tiempo_respuesta'].rolling(window=3, min_periods=1).std()
            
            # Métricas de aprendizaje
            participant_data['trial_normalizado'] = participant_data['Trial'] / participant_data['Trial'].max()
            participant_data['performance_trend'] = np.polyfit(range(len(participant_data)), participant_data['Distancia.t'], 1)[0]
            
            advanced_features.append(participant_data)
        
        # Combinar todos los datos
        self.df_advanced = pd.concat(advanced_features, ignore_index=True)
        
        print(f"Características creadas: {len([col for col in self.df_advanced.columns if col not in self.df.columns])}")
        print("Nuevas características:", [col for col in self.df_advanced.columns if col not in self.df.columns])
        
        return self.df_advanced
    
    def generate_comprehensive_report(self):
        """Generar reporte comprehensivo del análisis"""
        print("\n" + "="*60)
        print("           REPORTE COMPREHENSIVO DE ANÁLISIS COGNITIVO")
        print("="*60)
        
        # Resumen de datos
        print(f"\n📊 DATOS PROCESADOS:")
        print(f"   • Total de observaciones: {len(self.df):,}")
        print(f"   • Participantes únicos: {self.df['ID'].nunique():,}")
        print(f"   • Trials promedio por participante: {self.df.groupby('ID').size().mean():.1f}")
        
        # Resumen de modelos predictivos
        if self.models:
            print(f"\n🤖 MODELOS PREDICTIVOS:")
            for target, results in self.models.items():
                best_model = min(results.keys(), key=lambda x: results[x]['test_rmse'])
                best_rmse = results[best_model]['test_rmse']
                best_r2 = results[best_model]['test_r2']
                print(f"   • {target}:")
                print(f"     - Mejor modelo: {best_model}")
                print(f"     - RMSE: {best_rmse:.4f}")
                print(f"     - R²: {best_r2:.4f}")
        
        # Resumen de clustering
        if self.clusters_data:
            print(f"\n🧠 PERFILES COGNITIVOS:")
            optimal_k = self.clusters_data['optimal_k']
            print(f"   • Clusters identificados: {optimal_k}")
            
            profiles = self.clusters_data['profiles']
            for i in range(optimal_k):
                cluster_size = len(self.clusters_data['data'][self.clusters_data['data']['cluster'] == i])
                print(f"   • Cluster {i}: {cluster_size} participantes")
        
        # Recomendaciones
        print(f"\n💡 RECOMENDACIONES:")
        print("   • Implementar sistema de adaptación basado en perfiles cognitivos")
        print("   • Utilizar predicciones para personalizar dificultad del juego")
        print("   • Monitorear métricas de aprendizaje en tiempo real")
        
        return True

# Ejemplo de uso
if __name__ == "__main__":
    # Inicializar analizador
    analyzer = CognitiveBehaviorAnalyzer("datos_TFM.csv")
    
    # Pipeline completo de análisis
    print("Iniciando análisis cognitivo completo...")
    
    # 1. Exploración de datos
    analyzer.explore_data()
    
    # 2. Modelos predictivos
    analyzer.compare_predictive_models('Distancia.t')
    analyzer.visualize_model_performance('Distancia.t')
    
    # También para Error.t
    analyzer.compare_predictive_models('Error.t')
    analyzer.visualize_model_performance('Error.t')
    
    # 3. Clustering cognitivo
    analyzer.cognitive_clustering()
    analyzer.visualize_cognitive_clusters()
    analyzer.interpret_cognitive_profiles()
    
    # 4. Características avanzadas
    analyzer.advanced_feature_engineering()
    
    # 5. Reporte final
    analyzer.generate_comprehensive_report()
    
    print("\n🎉 Análisis completado exitosamente!")
