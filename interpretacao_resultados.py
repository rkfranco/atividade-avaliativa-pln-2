"""
Interpretação dos Resultados

Depois de passar dias rodando os códigos, coletando os dados e gerando gráficos, aqui está a nossa análise dos resultados das diferentes técnicas aplicadas ao nosso corpus:

== Bag of Words (BoW) ==
- O BoW ajudou a contar a frequência de palavras e identificar os termos mais comuns nos documentos. Foi interessante ver que o corpus tinha muitas palavras relacionadas a temas políticos e sociais. 
- Mas, sinceramente, BoW é meio limitado. Ele não entende o contexto das palavras, então é bom só para ter uma ideia geral das palavras mais usadas.

== TF-IDF ==
- O TF-IDF deu uma visão melhor sobre a importância das palavras nos documentos. Com ele, conseguimos ver que algumas palavras eram mais relevantes em documentos específicos. 
- No geral, a análise de similaridade cosseno mostrou que nossos documentos tinham temas bem variados, com poucas similaridades altas. Ótimo para mostrar que a base era diversificada, mas não ajuda muito a entender o contexto.

== Embeddings com SpaCy e SBERT ==
- Quando passamos para os embeddings, foi outro nível. O spaCy e o SBERT capturaram bem a similaridade semântica entre os textos. Com o SBERT, a busca semântica ficou bem precisa, mostrando os documentos mais parecidos com uma query que a gente dava.
- No fim das contas, embeddings foram os que mais mostraram o contexto real dos textos e ajudaram a fazer uma análise mais detalhada. 

== Clusters e PCA ==
- Aplicamos o K-Means e conseguimos visualizar os grupos de documentos com a ajuda do PCA em 3D. Foi legal ver como os documentos se agrupavam. Alguns clusters estavam bem separados, mas outros se misturavam, o que indica temas que meio que se cruzam.
- Essa parte foi boa para ver padrões, mas às vezes a separação dos clusters não era tão clara.

== Análise de Tópicos (LDA) ==
- A análise de tópicos com LDA foi útil para descobrir os principais assuntos do corpus. Vimos quais palavras apareciam mais em cada tópico e conseguimos ter uma noção dos temas discutidos.
- Mas é aquilo, interpretar esses tópicos requer um pouco mais de paciência, porque eles podem se sobrepor e nem sempre fazem sentido de primeira.

== Conclusão ==
Cada técnica teve seu momento de brilhar. BoW e TF-IDF foram ótimos para contar palavras e ver a importância delas, mas não ajudaram muito com o contexto. Quando precisávamos de algo mais preciso, os embeddings com spaCy e SBERT foram campeões, capturando a semântica dos textos. Os clusters e o PCA foram bons para ver agrupamentos, mas às vezes os temas se misturavam demais. Já o LDA ajudou a ter uma visão geral dos principais tópicos, mas a interpretação dos resultados foi meio trabalhosa.

No geral, conseguimos uma análise bem completa do corpus. Cada técnica trouxe uma peça do quebra-cabeça, e juntas nos ajudaram a entender melhor os padrões e a estrutura dos textos. Agora, vamos descansar porque, depois disso, nossa cabeça está a mil (3hrs manha n aguento mais de sono.

"""
