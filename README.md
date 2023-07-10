# Sentiment analysis API REST - FLASK  
_**Objectif:**_ Le but de ce challenge est d'industrialiser un notebook de Data Science et définir une architecture logicielle d’API.

**Mise en situation:** En tant que Machine Learning Engineer, un Data Scientist fait appel à vous afin de construire une API permettant de réaliser des inférences sur un modèle qu’il a entraîné.
***
### Preparation
1. Création de l'environement virtuelle Python 
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt 
    ```
2. Placer le dossier `out/` venant du Data Scientist à la racine du projet.  
3. Convertir le modèle
    ```bash
    python convert-model/convert.py --model_path="out/tf_model.h5" --config_path="out/config.json" --output_path="./tf-saved-model" 
    ```

---
### Tester localement le projet
> Note: Dans le cas d'un mac M1, il faut modifier le `docker-compose.yml` pour activer la bonne version de tf-server (image docker)
```bash
docker-compose build && docker-compose up -d 
```
### Déploiement en production
1. Construire l'image Docker
    ```bash
    docker build -t thekenken/challenge-soyhuce-mlops:latest .
   
    # Dans le cas d'un mac M1:
    docker buildx build --platform linux/amd64 -t thekenken/challenge-soyhuce-mlops:latest .
    ```

2. Push vers le docker hub 
    ```
    docker push thekenken/challenge-soyhuce-mlops:latest
    ```
3. Déploiement sur un cluster Kubernetes (non testé) 
    ```bash
    # Déploiement initial
    kubectl apply -f k8s_specifications/
    
    # Re-déploiement (Car le projet n'est pas versionné)
    kubectl rollout restart deployment sentiment-analysis
    kubectl rollout restart deployment tf-server
    ```

### Test
1. Executez le curl suivant pour la méthode bassique:
    ````bash
       curl -X 'POST' \
         'http://localhost:5001/v1/inference/sentiment_analysis' \
         -H 'accept: application/json' \
         -H 'Content-Type: application/json' \
         -d '{
         "sentence": "this is a bad example"
    }'
    ````
    **Resultat:**
    ```JSON
    {
         "sentiment": "NEGATIVE",
         "confidence": 0.98
    }
    ```
    
2. Executez le curl suivant pour l'approche tf-serving:
    ````bash
       curl -X 'POST' \
         'http://localhost:5001/v2/inference/sentiment_analysis' \
         -H 'accept: application/json' \
         -H 'Content-Type: application/json' \
         -d '{
         "sentence": "this is a bad example"
    }'
    ````
    **Resultat:**
    ```JSON
    {
         "sentiment": "NEGATIVE",
         "confidence": 0.98
    }
    ```

---
## Structure du projet

* `app.py`: le point d'entrée de l'application Flask qui définit les endpoints pour l'API.
* `sentiment_analysis.py`: le script qui charge le modèle pré-entraîné pour effectuer des prédictions de classification de sentiment.
* `requirements.txt`: les dépendances Python requises pour exécuter l'application Flask.
* `Dockerfile`: le fichier Dockerfile utilisé pour construire l'image Docker.
* `docker-compose.yml`: le fichier de configuration Docker Compose utilisé pour créer et exécuter les conteneurs Docker.
* `out/`: le dossier contenant le modèle pré-entraîné, y compris les fichiers de configuration, les poids et les tokens.
* `k8s_specifications/`: le dossier contenant les spécifications K8s pour le déploiement de l'application
* `wait_for_tfserver.py`: script qui check si le conteneur/serveur tf-server est bien démarré.
* `tests/`: Tests unitaires

***
## Les choix d'implémentation technique et d'architecture du projet
### - Techniques
* `Flask` et `Docker`: Utilisé pendant mon stage de fin études et mon projet de fin études.
* `Kubernetes`: Faciliter le déploiement dans un service cloud

### Architecture - Basique 
![architecture_docker.png](img%2Farchitecture_docker.png)
Utilisation d'un Serveur `Flask` et chargement du model dans ce serveur.
 * **Pros:**
   * Déploiement facile 
   * Approche simple
 * **Cons:**
   * Lourdeur - Nécessite d'installer des packages lourds dans l'image Docker
   * La gestion de plusieurs versions du modèles est manuelle est donc risqué.
   * Problème de stabilité: Si tensorflow plante, le serveur Flask est indisponible 


### Architecture - tf-serving 
![architecture_tf-serving.png](img%2Farchitecture_tf-serving.png)
Utilisation d'un serveur `Flask` devant un serveur `tf-serving`. 

Dans cette approche, le modèle est seulement déployé dans le serveur `tf-server`. Le serveur `Flask` va effectuer un HTTP POST request vers `tf-server` pour récupérer le résulat de la prédiction

 * **Pros:** 
   * Gestion simple du déploiement de version multiple du modèle grâce à `tensorflow/serving`
   * Séparation forte entre l'API public et le modèle ML
   * Le déploiement du serveur Flask est plus leger
   * Stabilité du service 
 * **Cons:** 
   * Besoin de sauvegarder le modèle pré-entraîné en utilisant le format *.pb
   * `tensorflow/serving` - Apprentissage technique 

---
## Les possiblités d'amélioration
* Améliorer le temps de démarrage du serveur Flask (Probablement lié au démarrage de tensorflow)
* Optimiser la taille des images docker 
* Contacter le data scientist pour modifier le notebook et enregistrer le modèle dans le bon format
* Améliorer le health_check.py
* Ajouter la notion de version dans le projet (remplacer le tag `latest` par une version)
* Améliorer la gestion d'erreur
* Ajouter des tests unitaires
* Approfondir tensorflow/serving pour améliorer la scabilité, gestion de la mémoire

---
## Issues
* <u>**Problème tf-serving:**</u>
  * _Problème d'architecture._ L'image `tensorflow/serving` ne fonctionne pas sur une ARM64. 
  * _Solution:_ 
    * Utiliser une autre image compatible avec: `emacski/tensorflow-serving:latest-linux_arm64`
    * Déployer sur une autre machine. Utilisation de cloud.
* <u>**Problème compatibilité du model et tf-serving:**</u>
  * ```docker
    No versions of servable sentiment_analysis found under base path /models/sentiment_analysis/1. Did you forget to name your leaf directory as a number (eg. '/1/')?
    ```
  * Answer [here](https://stackoverflow.com/questions/45544928/tensorflow-serving-no-versions-of-servable-model-found-under-base-path) - The model is not saved in the correct format 
  * On a besoin d'un fichier "saved_model.pb" + d'un dossier "variables"
  * Solution : [ici](./convert-model/convert.py)
* <u>**Response tf-server - ERROR 400**</u> 
  * ```
    {'error': 'In[0] should be a scalar: [6]\n\t [[{{node tf_bert_for_sequence_classification/bert/embeddings/assert_less/Assert/Assert}}]]'}
    ```
  * Pas eu le temps de régler ce problème
* <u>**Problème Docker:**</u> 
```docker 
Collecting zipp>=0.5
#8 105.7   Downloading zipp-3.15.0-py3-none-any.whl (6.8 kB)
#8 105.8 Building wheels for collected packages: jax, psutil
#8 105.8   Building wheel for jax (pyproject.toml): started
#8 106.3   Building wheel for jax (pyproject.toml): finished with status 'done'
#8 106.3   Created wheel for jax: filename=jax-0.4.8-py3-none-any.whl size=1439678 sha256=d37d4df915f1eb19f3a604aef6ba7c905239b07640c39b3318a2b4192ae0ff7b
#8 106.3   Stored in directory: /tmp/pip-ephem-wheel-cache-z5xtexgu/wheels/05/94/dc/81042da9bced43ff430bc02043d213d9e4b210b584c39e31c1
#8 106.3   Building wheel for psutil (pyproject.toml): started
#8 106.9   Building wheel for psutil (pyproject.toml): finished with status 'error'
#8 107.0   error: subprocess-exited-with-error
#8 107.0   
#8 107.0   × Building wheel for psutil (pyproject.toml) did not run successfully.
#8 107.0   │ exit code: 1
#8 107.0   ╰─> [43 lines of output]
#8 107.0       running bdist_wheel
#8 107.0       running build
#8 107.0       running build_py
#8 107.0       creating build
#8 107.0       creating build/lib.linux-aarch64-cpython-39
#8 107.0       creating build/lib.linux-aarch64-cpython-39/psutil
#8 107.0       copying psutil/_psbsd.py -> build/lib.linux-aarch64-cpython-39/psutil
#8 107.0       copying psutil/__init__.py -> build/lib.linux-aarch64-cpython-39/psutil
#8 107.0       copying psutil/_compat.py -> build/lib.linux-aarch64-cpython-39/psutil
#8 107.0       copying psutil/_pswindows.py -> build/lib.linux-aarch64-cpython-39/psutil
#8 107.0       copying psutil/_common.py -> build/lib.linux-aarch64-cpython-39/psutil
#8 107.0       copying psutil/_psposix.py -> build/lib.linux-aarch64-cpython-39/psutil
#8 107.0       copying psutil/_pslinux.py -> build/lib.linux-aarch64-cpython-39/psutil
#8 107.0       copying psutil/_psosx.py -> build/lib.linux-aarch64-cpython-39/psutil
#8 107.0       copying psutil/_psaix.py -> build/lib.linux-aarch64-cpython-39/psutil
#8 107.0       copying psutil/_pssunos.py -> build/lib.linux-aarch64-cpython-39/psutil
#8 107.0       creating build/lib.linux-aarch64-cpython-39/psutil/tests
#8 107.0       copying psutil/tests/test_contracts.py -> build/lib.linux-aarch64-cpython-39/psutil/tests
#8 107.0       copying psutil/tests/test_windows.py -> build/lib.linux-aarch64-cpython-39/psutil/tests
#8 107.0       copying psutil/tests/test_testutils.py -> build/lib.linux-aarch64-cpython-39/psutil/tests
#8 107.0       copying psutil/tests/__init__.py -> build/lib.linux-aarch64-cpython-39/psutil/tests
#8 107.0       copying psutil/tests/test_osx.py -> build/lib.linux-aarch64-cpython-39/psutil/tests
#8 107.0       copying psutil/tests/test_posix.py -> build/lib.linux-aarch64-cpython-39/psutil/tests
#8 107.0       copying psutil/tests/test_connections.py -> build/lib.linux-aarch64-cpython-39/psutil/tests
#8 107.0       copying psutil/tests/test_aix.py -> build/lib.linux-aarch64-cpython-39/psutil/tests
#8 107.0       copying psutil/tests/test_sunos.py -> build/lib.linux-aarch64-cpython-39/psutil/tests
#8 107.0       copying psutil/tests/test_misc.py -> build/lib.linux-aarch64-cpython-39/psutil/tests
#8 107.0       copying psutil/tests/test_system.py -> build/lib.linux-aarch64-cpython-39/psutil/tests
#8 107.0       copying psutil/tests/test_unicode.py -> build/lib.linux-aarch64-cpython-39/psutil/tests
#8 107.0       copying psutil/tests/test_bsd.py -> build/lib.linux-aarch64-cpython-39/psutil/tests
#8 107.0       copying psutil/tests/runner.py -> build/lib.linux-aarch64-cpython-39/psutil/tests
#8 107.0       copying psutil/tests/test_process.py -> build/lib.linux-aarch64-cpython-39/psutil/tests
#8 107.0       copying psutil/tests/__main__.py -> build/lib.linux-aarch64-cpython-39/psutil/tests
#8 107.0       copying psutil/tests/test_linux.py -> build/lib.linux-aarch64-cpython-39/psutil/tests
#8 107.0       copying psutil/tests/test_memleaks.py -> build/lib.linux-aarch64-cpython-39/psutil/tests
#8 107.0       running build_ext
#8 107.0       building 'psutil._psutil_linux' extension
#8 107.0       creating build/temp.linux-aarch64-cpython-39
#8 107.0       creating build/temp.linux-aarch64-cpython-39/psutil
#8 107.0       gcc -pthread -Wno-unused-result -Wsign-compare -DNDEBUG -g -fwrapv -O3 -Wall -fPIC -DPSUTIL_POSIX=1 -DPSUTIL_SIZEOF_PID_T=4 -DPSUTIL_VERSION=594 -DPy_LIMITED_API=0x03060000 -DPSUTIL_LINUX=1 -DPSUTIL_ETHTOOL_MISSING_TYPES=1 -I/usr/local/include/python3.9 -c psutil/_psutil_common.c -o build/temp.linux-aarch64-cpython-39/psutil/_psutil_common.o
#8 107.0       C compiler or Python headers are not installed on this system. Try to run:
#8 107.0       sudo apt-get install gcc python3-dev
#8 107.0       error: command 'gcc' failed: No such file or directory
#8 107.0       [end of output]
#8 107.0   
#8 107.0   note: This error originates from a subprocess, and is likely not a problem with pip.
#8 107.0 Successfully built jax
#8 107.0 Failed to build psutil
#8 107.0   ERROR: Failed building wheel for psutil
#8 107.0 ERROR: Could not build wheels for psutil, which is required to install pyproject.toml-based projects
#8 107.1 WARNING: You are using pip version 22.0.4; however, version 23.1 is available.
#8 107.1 You should consider upgrading via the '/usr/local/bin/python -m pip install --upgrade pip' command.
------
executor failed running [/bin/sh -c pip install --no-cache-dir -r requirements.txt]: exit code: 1
```
  * _Problème_: utilisation de `FROM python:3.9-slim` et installation de packages qui ont besoin d'un Python 'complet'
  * _Solution_: 
    * Utiliser un `python:3.9`
    * Nettoyer: Enlever tous les packages non utilisés et non essentiels à l'API 
*** 
* Problème Docker vs ARM64
```docker
invalid character 'c' looking for beginning of value
```
![error_docker.png](img%2Ferror_docker.png)
Je n'ai pas très bien compris ce problème. Il semblerait qu'il y ait eu des soucis entre docker et ma machine.

_Solution_ _j'ai dû re-installer docker._   