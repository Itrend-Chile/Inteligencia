README

Este repositorio contiene todos los códigos que se han utilizado para distintos análisis que ha hecho el área de Inteligencia de Itrend
Están organizados en las siguientes carpetas

1) 01_Extraccion: 
	Códigos para extraer automáticamente información de publicaciones científicas de SCOPUS, usando key words
	Contiene los siguientes códigos:

	01_Scopus.py: código python para obtener los papers a partir de la libreta de académicos (Base_academicos.xlsx) que está en la carpeta de base de datos

	02_Text cleaning.py: código python para limpiar los datos de la base de datos de papers descargada a partir 01_Scopus.py

	03_Scopus_keywords.py: código python para buscar papers en Scopus a partir de palabras clave asociadas a distintas amenazas. [Importante: este es el código más utilizado para extracción de Scopus]


2) 02_Analisis_texto

	05_Pre proceso VF.py: código python que hace una limpieza de la base de datos de articulos académicos extraída con el código 03_Scopus_keywords.py. Además general un modelo de tópicos para apoyar a la clasificación manual

	06_Clasificacion_supervisada: código python que utiliza la base output de 05_Pre proceso VF.py y una base de datos etiquetada (labels.xlsx), que se encuentra en la carpeta de bases de datos. El programa genera un modelos de clasificación supervisada para clasificar los articulos por amenaza.

	10_Topicos_nota_tecnica.py: código python que utiliza la base de datos etiquetada (Clas_auto_manual_v2.xlsx) para generar modelos de tópicos para 3 amenazas geológicas: terremotos, tsunami y volcanes. Este código se utilizó para la nota https://conectaresiliencia.cl/los-estudios-de-cientificos-chilenos-sobre-terremotos-quintuplican-los-estudios-sobre-tsunamis/
	Además el output de este código se utiliza para generar las visualizaciones de la nota, que se generan con el código scatterpie_topicos.R

	12_Topicos_nota_incendios.py: código python q ue utiliza la base de datos etiquetada (Clas_auto_manual_v3.xlsx) para generar modelos de tópicos para articulos en incendios forestales.

3) 03_Otros_analisis
	Esta carpeta contiene códigos para transformar la información de la base de artículos a autores o afiliaciones, por amenaza

4) 04_Visualizaciones
	Contiene los códigos para generar distintos tipo de análisis gráficos

5) Bases de datos
	Contiene bases de datos que sirven de input para algunos de los códigos ya detallados anteriormente



