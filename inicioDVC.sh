# 1. Inicializar Git y DVC
git init
dvc init

# 2. Ignorar carpetas temporales y artefactos pesados en Git
echo -e "mlruns/\nlogs/\nmodels/*.joblib\n__pycache__/\n.venv/" >> .gitignore

# 3. Ejecutar el pipeline por primera vez con DVC (crea dvc.lock real)
dvc repro

# 4. Guardar la Versión 1 de los datos en Git y en la caché de DVC
git add .
git commit -m "v1: Dataset inicial validado (600 filas)"
git tag -a "datos-v1" -m "Versión 1 auditada del dataset"

# 5. Entrenar el modelo con datos-v1
python train.py