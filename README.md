
# Decifrando a Caixa Preta: Tornando Modelos de IA Explicáveis com LIME

Com o avanço das tecnologias de inteligência artificial, modelos preditivos têm sido utilizados em diversos setores, incluindo o financeiro. Sistemas automatizados são empregados para avaliar o risco de concessão de crédito em bancos, com base em características dos clientes, como idade, renda, histórico de inadimplência, entre outras variáveis. Embora esses modelos apresentem bons índices, muitas decisões geradas por eles são de difícil compreensão para os usuários finais, gestores e órgãos reguladores.

A necessidade de tornar os modelos mais compreensíveis impulsionou o desenvolvimento de técnicas voltadas à interpretabilidade, reunidas sob o campo da Inteligência Artificial Explicável, XAI (Explainable Artificial Intelligence). O objetivo dessas técnicas é fornecer justificativas para as previsões realizadas pelos algoritmos, promovendo maior confiança, transparência e aderência a requisitos éticos e regulatórios.

Este trabalho tem como propósito aplicar uma técnica de interpretabilidade baseada na biblioteca LIME, a fim de explicar decisões individuais de um modelo de classificação utilizado na análise de crédito. Para isso, será utilizado um conjunto de dados reconhecido, contendo informações de clientes categorizados como bons ou maus pagadores. Ao final, espera-se compreender como características específicas influenciam diretamente nas decisões do modelo e de que forma essas explicações podem ser utilizadas.


## Contextualização do problema e objetivos

A concessão de crédito bancário é uma atividade que envolve riscos, tanto para instituições financeiras quanto para os clientes. Tradicionalmente, as decisões de aprovação de crédito eram baseadas na análise humana e no julgamento subjetivo de analistas. Com o surgimento de grandes volumes de dados e algoritmos de aprendizado de máquina, esse processo passou a ser automatizado, visando maior eficiência e padronização nas decisões. Apesar de os modelos preditivos modernos apresentarem bom desempenho em tarefas de classificação, como identificar clientes com maior ou menor risco de inadimplência, eles tendem a operar como caixas-pretas, que significa que produzem resultados sem que o processo seja compreendido. Essa característica dificulta a aceitação por parte dos clientes e a validação por entidades reguladoras, especialmente em decisões negativas.

Assim, este trabalho propõe a utilização de técnicas de compreensão, com ênfase em LIME para explicar de forma individual as decisões tomadas por um modelo de classificação aplicado ao problema de crédito. O objetivo é tornar o processo mais transparente, identificando quais atributos de cada cliente tiveram maior impacto na classificação como bom ou mau pagador. Os objetivos são:

- Desenvolver um modelo preditivo de classificação binária utilizando um conjunto de dados de crédito real;
- Integrar a biblioteca LIME ao fluxo de decisão do modelo para gerar explicações locais;
- Visualizar e interpretar os fatores que influenciam as previsões do modelo para instâncias específicas;
- Discutir os resultados obtidos 
## Modelo preditivo explicado

Para realizar a classificação dos clientes como sendo de bom ou mau risco de crédito, foi implementado um modelo baseado no algoritmo Random Forest, um método de aprendizado de máquina que utiliza o princípio de ensemble de árvores de decisão. Este algoritmo foi escolhido por sua robustez, capacidade de generalização, facilidade no uso e bom desempenho com dados que apresentam atributos numéricos e categóricos.

Como etapa inicial, foi feito o carregamento do conjunto dos dados Statlog sugeridos pela faculdade. O dataset contém 1.000 instâncias, cada uma com 20 atributos. A variável alvo, chamada de `credit_risk` no código, indica se o cliente representa um risco bom (1) ou mau (0).

Para garantir um pré-processamento adequado, foi utilizado um Pipeline, que automatiza o tratamento das variáveis antes de aplicar o modelo. O pipeline utilizado é composto pelas seguintes etapas:

1. Separação entre variáveis numéricas e categóricas;
2. Padronização das variáveis numéricas com StandardScaler;
3. Codificação one-hot das variáveis categóricas com OneHotEncoder;
4. Aplicação do classificador RandomForestClassifier.
 
 ㅤ

```python
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

X = df.drop(columns=['Target'])
y = df['Target']
categorical_cols = X.select_dtypes(include='object').columns.tolist()
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

preprocessor = ColumnTransformer(transformers=[
    ('num', StandardScaler(), numerical_cols),
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
])

pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'))
])
```

Após a construção do pipeline, o modelo foi treinado com um conjunto de treinamento derivado da divisão dos dados. A métrica de avaliação principal foi a acurácia, além de precisão, revocação e f1-score, conforme o relatório de classificação gerado.

```python
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)
print("---Relatório de Classificação---\n", (classification_report(y_test, y_pred)))
print("---Matrix de Confusão---\n", (confusion_matrix(y_test, y_pred)))
```
![Resultado do pipeline](https://github.com/CKOArquiteto/Explainable-AI/blob/main/Resultados%20e%20Imagens/Resultados%20de%20código%20fonte.png) 


O uso do parâmetro `class_weight='balanced'` no classificador tem por objetivo compensar possíveis desbalanceamentos entre as classes, ajustando os pesos de penalização de acordo com a frequência dos rótulos. Apesar de o modelo inicial não apresentar uma acurácia elevada, seu uso é fundamental para a próxima etapa do projeto: a explicação das previsões individuais com a técnica LIME.



## Contextualização do problema e objetivos

A concessão de crédito bancário é uma atividade que envolve riscos, tanto para instituições financeiras quanto para os clientes. Tradicionalmente, as decisões de aprovação de crédito eram baseadas na análise humana e no julgamento subjetivo de analistas. Com o surgimento de grandes volumes de dados e algoritmos de aprendizado de máquina, esse processo passou a ser automatizado, visando maior eficiência e padronização nas decisões. Apesar de os modelos preditivos modernos apresentarem bom desempenho em tarefas de classificação, como identificar clientes com maior ou menor risco de inadimplência, eles tendem a operar como caixas-pretas, que significa que produzem resultados sem que o processo seja compreendido. Essa característica dificulta a aceitação por parte dos clientes e a validação por entidades reguladoras, especialmente em decisões negativas.

Assim, este trabalho propõe a utilização de técnicas de compreensão, com ênfase em LIME para explicar de forma individual as decisões tomadas por um modelo de classificação aplicado ao problema de crédito. O objetivo é tornar o processo mais transparente, identificando quais atributos de cada cliente tiveram maior impacto na classificação como bom ou mau pagador. Os objetivos são:

- Desenvolver um modelo preditivo de classificação binária utilizando um conjunto de dados de crédito real;
- Integrar a biblioteca LIME ao fluxo de decisão do modelo para gerar explicações locais;
- Visualizar e interpretar os fatores que influenciam as previsões do modelo para instâncias específicas;
- Discutir os resultados obtidos 
## As explicações geradas pelo LIME

O uso da biblioteca LIME (Local Interpretable Model-agnostic Explanations) neste projeto teve como principal finalidade gerar explicações locais para decisões específicas do modelo de classificação. Em outras palavras, LIME permite entender, para cada cliente analisado, quais características mais influenciaram a decisão de aprovar ou negar o crédito.

Ao aplicar o LimeTabularExplainer, a explicação gerada refere-se a um único exemplo do conjunto de dados. O LIME perturba ligeiramente essa instância, criando variações próximas dela e avalia como o modelo responde a essas perturbações. A seguir, é ajustado um modelo linear simples para aproximar o comportamento do classificador complexo naquela região do espaço de entrada. A chamada principal para a explicação é:

```python
exp = explainer.explain_instance(
    X_test_transformed[i],
    model.predict_proba,  
    num_features=10
)
```
O resultado é uma lista de pares, indicando o efeito local de cada atributo na probabilidade de classificação da instância. Um peso positivo indica que a presença daquela característica aumentou a chance de o cliente ser classificado como bom risco, enquanto um peso negativo contribuiu para a classificação como mau risco. A visualização é feita com:
```python
fig = exp.as_pyplot_figure()
fig.savefig('explicacoes cliente 10.png', bbox_inches='tight')
plt.show()
```

### Interpretando o exemplo
Para o cliente utilizado do conjunto de testes somos capazes de analizar e interpretar os dados apresentados, podendo verificar se o modelo realmente é capaz de ser confiado em decisões financeiras. A explicação gerada indicou:

- O atributo `duration_in_month` com valor alto contribuiu negativamente para a aprovação; 
- O atributo `checking_account_status` com valor no checking também teve impacto negativo;
- Por outro lado, a presença de um bom `credit_history` e savings_account com saldo elevado contribuiu positivamente para a classificação como bom pagador.

Essa visualização permite que analistas humanos compreendam quais fatores individuais mais influenciaram a decisão, o que pode ser comunicado ao cliente ou utilizado para fins de auditoria regulatória. As explicações fornecidas pelo LIME são particularmente úteis porque traduzem decisões complexas de modelos como Random Forests em termos compreensíveis.
## Limitações do modelo LIME
A interpretabilidade de modelos de aprendizado de máquina é um aspecto cada vez mais relevante em aplicações de alto impacto social, como a concessão de crédito. Embora modelos complexos, como florestas aleatórias e redes neurais, alcancem alta acurácia, sua natureza opaca torna difícil compreender os critérios utilizados nas decisões. Isso compromete não apenas a confiança do usuário final, mas também o cumprimento de exigências regulatórias em setores como o financeiro.

A adoção da técnica LIME neste trabalho demonstrou ser uma ferramenta valiosa para abrir a “caixa-preta” do modelo. Ao produzir explicações locais para previsões individuais, foi possível identificar quais atributos mais influenciaram a decisão de classificar um cliente como bom ou mau pagador. Essa abordagem trouxe diversos benefícios, entre eles:

- Transparência no processo decisório, permitindo que clientes e analistas entendam os principais motivos de uma recusa ou aprovação;
- Possibilidade de auditoria do modelo por entidades reguladoras, com explicações técnicas e visualmente acessíveis;
- Identificação de padrões e possíveis injustiças no modelo, como impacto desproporcional de variáveis demográficas ou socioeconômicas;
- Aprimoramento do modelo, possibilitando uma análise crítica sobre variáveis com influência indevida ou limitada relevância.

Apesar de sua utilidade, o uso do LIME apresenta algumas limitações importantes:

- Explicações instáveis: pequenas alterações na instância analisada podem gerar diferentes explicações, o que pode confundir usuários leigos;
- Simplificações locais: o modelo explicativo do LIME é linear, o que pode não representar bem regiões mais complexas da função de decisão;
- Necessidade de pré-processamento consistente: como o LIME opera sobre os dados de entrada, é essencial garantir que as mesmas transformações aplicadas ao treino sejam corretamente invertidas ou replicadas para a explicação.

No presente trabalho, mesmo com esses cuidados, observou-se que a estabilidade e a clareza das explicações variaram conforme a instância analisada e o desempenho geral do modelo. Ainda assim, os resultados obtidos permitiram uma compreensão muito mais acessível do funcionamento interno do classificador, cumprindo com sucesso os objetivos propostos no início do projeto. A interpretabilidade não deve ser vista como uma etapa opcional, mas como parte integrante e indispensável no ciclo de desenvolvimento de soluções baseadas em inteligência artificial, especialmente em contextos sensíveis como o financeiro.
## Considerações finais

Os resultados demonstraram que, embora o modelo de classificação selecionado possua desempenho moderado, sua integração com a interpretabilidade possibilitou compreender, de forma clara e fundamentada, os fatores que influenciam cada decisão individual. Essa capacidade de explicação é especialmente relevante em contextos regulados e de alta responsabilidade, como o setor financeiro, em que decisões automatizadas precisam ser justificadas para clientes, gestores e órgãos fiscalizadores.

A técnica LIME se mostrou eficaz ao fornecer explicações compreensíveis e visualmente intuitivas, auxiliando na identificação de variáveis mais impactantes nas classificações. No entanto, foram observadas limitações, como a sensibilidade a pequenas variações nos dados e a dependência da coerência no pré-processamento.

De modo geral, este estudo reforça que a interpretabilidade deve ser considerada elemento essencial no ciclo de desenvolvimento de modelos de inteligência artificial, tanto para garantir transparência e conformidade regulatória quanto para promover confiança e equidade no uso dessas tecnologias. A continuidade desse trabalho pode incluir ajustes de hiperparâmetros, análise de desempenho com outros algoritmos e comparação com técnicas alternativas de interpretabilidade, como SHAP, visando ampliar a robustez e a utilidade das explicações.
