![GitHub repo size](https://img.shields.io/github/repo-size/marcelagomescorrea/analisando-e-recriando-fotos-candidatos-eleicoes-brasil)
![GitHub language count](https://img.shields.io/github/languages/count/marcelagomescorrea/analisando-e-recriando-fotos-candidatos-eleicoes-brasil)

# Analisando e recriando foto de candidatos eleitos nas últimas eleições no Brasil

> Busca por padrões em fotos utilizando a Análise de Componentes Principais (PCA) e recriação dessas fotos a partir do uso de Convolutional Neural Network (CNN) com Deep Learning.

## ☕ Descrição do problema

O projeto atual teve como ponto de partida os questionamentos provocados no artigo da Folha de São Paulo intitulado [Mulheres com traços considerados mais masculinos têm vantagem em eleições no Brasil](https://www1.folha.uol.com.br/ciencia/2022/10/mulheres-com-tracos-considerados-mais-masculinos-tem-vantagem-em-eleicoes-no-brasil-aponta-estudo.shtml). Segundo estudo apresentado no artigo, o rosto de candidatas mulheres tem influência sobre o desempenho das mesmas nas eleições, quanto mais masculino os traços, maiores as chances de serem eleitas.

A partir daí, esse projeto se propõe a analisar padrões em fotos dos candidatos eleitos nas últimas eleições no Brasil (2022 a 2014) utilizando tanto técnicas de aprendizado não supervisionado (unsupervised learning) quanto de aprendizado profundo (deep learning).

## 🚀 Solução de IA

Como etapa inicial, utilizamos o PCA (Análise de Componentes Principais) para verificar se de fato há padrões na foto do candidato padrão eleito e comparamos com o não eleito. Em seguida criamos uma rede neural com arquitetura autoencoder, mais especificamente autoencoder variacional (VAE), que lida melhor na geração de novas imagens.

## 📫 Fonte dos dados
Para treinamento do modelo e da rede neural, foram utilizados fotos de candidatos cadastrados no sistema do TSE das eleições de 2022 a 2014, sendo que a maioria das fotos até 2016 são coloridas e em 2014 são em preto e branco. Também foi utilizado arquivos .csv contendo informações dos candidatos que foram eleitoes ou não.

Os dados utilizados estão disponíveis no [Portal de Dados Abertos](https://dadosabertos.tse.jus.br/) do Tribunal Superior Eleitoral (TSE). 


# Startup the project (em construção)

The initial setup.

Create virtualenv and install the project:
```bash
sudo apt-get install virtualenv python-pip python-dev
deactivate; virtualenv ~/venv ; source ~/venv/bin/activate ;\
    pip install pip -U; pip install -r requirements.txt
```

Unittest test:
```bash
make clean install test
```

Check for analisando-e-recriando-fotos-candidatos-eleicoes-brasil in github.com/{group}. If your project is not set please add it:

Create a new project on github.com/{group}/analisando-e-recriando-fotos-candidatos-eleicoes-brasil
Then populate it:

```bash
##   e.g. if group is "{group}" and project_name is "analisando-e-recriando-fotos-candidatos-eleicoes-brasil"
git remote add origin git@github.com:{group}/analisando-e-recriando-fotos-candidatos-eleicoes-brasil.git
git push -u origin master
git push -u origin --tags
```

Functionnal test with a script:

```bash
cd
mkdir tmp
cd tmp
analisando-e-recriando-fotos-candidatos-eleicoes-brasil-run
```

# Install

Go to `https://github.com/{group}/analisando-e-recriando-fotos-candidatos-eleicoes-brasil` to see the project, manage issues,
setup you ssh public key, ...

Create a python3 virtualenv and activate it:

```bash
sudo apt-get install virtualenv python-pip python-dev
deactivate; virtualenv -ppython3 ~/venv ; source ~/venv/bin/activate
```

Clone the project and install it:

```bash
git clone git@github.com:{group}/analisando-e-recriando-fotos-candidatos-eleicoes-brasil.git
cd analisando-e-recriando-fotos-candidatos-eleicoes-brasil
pip install -r requirements.txt
make clean install test                # install and test
```
Functionnal test with a script:

```bash
cd
mkdir tmp
cd tmp
analisando-e-recriando-fotos-candidatos-eleicoes-brasil-run
```
