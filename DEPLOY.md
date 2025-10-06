# 🚀 Guia de Deploy - Streamlit Cloud

## ✅ Checklist Pré-Deploy

### 1. Arquivos Configurados
- [x] `app.py` - Carrega recursos do GitHub
- [x] `requirements.txt` - Todas as dependências listadas
- [x] `.streamlit/config.toml` - Tema dark premium configurado
- [x] `.gitignore` - Configurado corretamente
- [x] `README.md` - Instruções de deploy atualizadas

### 2. URLs do GitHub Configuradas
O app está configurado para carregar automaticamente do repositório:
```
https://github.com/sidnei-almeida/secom_failure_prediction
```

**Arquivos carregados automaticamente:**
- 📊 `data/secom_cleaned_dataset.csv`
- 🧠 `models/secom_autoencoder_model.keras`
- 📝 `training/secom_autoencoder_metadata.json`

### 3. Dependências Necessárias
```
✓ streamlit>=1.28.0
✓ streamlit-option-menu>=0.3.6
✓ tensorflow-cpu>=2.15.0
✓ pandas>=2.0.0
✓ numpy>=1.24.0
✓ scikit-learn>=1.3.0
✓ plotly>=5.17.0
✓ Pillow>=10.0.0
✓ requests>=2.31.0
```

## 📤 Passos para Deploy

### 1. Commit e Push para GitHub
```bash
# Adicionar todos os arquivos (incluindo data/, models/, training/)
git add .

# Commit
git commit -m "Deploy: App pronto para Streamlit Cloud"

# Push para main
git push origin main
```

### 2. Deploy no Streamlit Cloud

1. Acesse: [share.streamlit.io](https://share.streamlit.io)
2. Faça login com GitHub
3. Clique em "New app"
4. Selecione:
   - **Repository**: `sidnei-almeida/secom_failure_prediction`
   - **Branch**: `main`
   - **Main file path**: `app.py`
5. Clique em "Deploy!"

### 3. Aguarde o Build
O Streamlit Cloud irá:
- Instalar as dependências do `requirements.txt`
- Carregar os arquivos do GitHub automaticamente
- Aplicar o tema do `.streamlit/config.toml`
- Iniciar o app

⏱️ Tempo estimado: 3-5 minutos

## 🎨 Recursos do App

### Páginas
1. **🏠 Home** - Overview e métricas principais
2. **📊 Análise de Dados** - Exploração do dataset SECOM
3. **🧠 Modelo** - Arquitetura do Autoencoder
4. **📈 Treinamento** - Histórico e performance
5. **🔬 Teste** - Detecção de anomalias em tempo real

### Design
- 🌑 Tema dark premium
- 🔥 Paleta de cores quente (laranja/fogo)
- ✨ Efeitos visuais elegantes (glows, shadows)
- 📱 Layout responsivo

### Thresholds de Detecção
- **Balanced (0.45)**: Equilíbrio entre precision e recall
- **Conservative (0.50)**: Menos falsos positivos

## 🔧 Troubleshooting

### Erro ao carregar dados
- Verifique se os arquivos estão commitados no GitHub
- Confirme que o repositório está público ou que o Streamlit Cloud tem acesso
- Branch deve ser `main`

### Erro de dependências
- Verifique `requirements.txt`
- TensorFlow CPU é usado para compatibilidade

### Erro de tema
- Arquivo `.streamlit/config.toml` deve estar no repositório
- Não deve estar no `.gitignore`

## 📞 Suporte

- [Documentação Streamlit Cloud](https://docs.streamlit.io/streamlit-community-cloud)
- [Fórum Streamlit](https://discuss.streamlit.io/)

---

**✨ Pronto para deployment!** O app está 100% configurado para rodar no Streamlit Cloud sem nenhuma configuração adicional.

