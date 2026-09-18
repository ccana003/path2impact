# Path2Impact

**Path2Impact** is an AI-assisted research tool developed as part of my Master's thesis in the **Master of Science in Clinical and Translational Investigation (MSCTI)** program at the University of Miami.

The project was created to explore a challenging question in translational science:

> **Can we use artificial intelligence to help measure how research changes over time and whether it reflects key principles of translational science?**

## About the Project

The Clinical and Translational Science Awards (CTSA) Program is designed to accelerate the translation of scientific discoveries into improvements in human health. However, measuring that impact is difficult.

Traditional approaches such as citation counts can measure the dissemination of research, while frameworks such as the T1–T4 translational spectrum can describe where research falls along the translational continuum. These approaches do not necessarily capture *how* translational science is being practiced within the research itself.

Path2Impact was developed as an exploratory approach to this problem.

Rather than relying only on traditional bibliometric measures, the tool uses a Large Language Model (LLM) to analyze the content of scientific publications and evaluate the presence of selected **NCATS Principles of Translational Science**.

## How Path2Impact Works

Path2Impact provides a web-based interface for submitting scientific publications for analysis.

The application extracts publication content and sends structured prompts to an LLM that evaluates the research according to predefined translational science criteria.

For each publication, the system:

1. Processes the publication content.
2. Evaluates the research against selected NCATS Principles of Translational Science.
3. Assigns structured scores based on predefined scoring criteria.
4. Returns the results to the application.
5. Stores the results for subsequent analysis.

This creates structured data that can be analyzed across groups of publications rather than evaluating research impact solely through citation counts or other traditional bibliometric measures.

## Master's Thesis Study

Path2Impact served as the technical platform for my MSCTI capstone/thesis research.

For the study, publications associated with the University of Miami were evaluated from periods **before and after CTSA implementation**.

The AI-generated assessments were then compared with evaluations performed by human reviewers to determine how closely the automated approach aligned with expert assessment.

The study was designed as an initial validation of whether LLMs could reliably identify and quantify characteristics of translational science within scientific publications.

## Why I Built It

Path2Impact began as an attempt to connect two areas that interested me: **translational science and research technology**.

Instead of asking only how many publications were produced or how frequently they were cited, I wanted to explore whether the publications themselves could tell us something about how translational research was being conducted.

The project ultimately became both a research study and a working software prototype demonstrating how AI-assisted analysis could potentially contribute to the evaluation of translational science.

## Important Note

Path2Impact was developed as a **research prototype for an academic master's thesis**.

It should not be interpreted as a validated replacement for expert review or as an official evaluation tool of the University of Miami, NCATS, NIH, or the CTSA Program.

Rather, the project demonstrates a potential approach for using LLMs to transform qualitative characteristics of scientific publications into structured data that can be studied quantitatively.

## Author

**Carlos A. Canales**  
Master of Science in Clinical and Translational Investigation  
University of Miami
