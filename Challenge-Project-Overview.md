---

> ## Challenge Advisor: Update & Finalize Your Project Overview
>
> > 💡 **These grey text instructions are just for you, the team's Challenge Advisor; please delete them once you have completed the steps below.**
>
> We've pre-populated this Challenge Project Overview page — which is what will be shared with your Break Through Tech student team in August — using the details from your submission form. You should have received an email inviting you to join this repo as a Collaborator, enabling you to add files and make edits.
> 
> In order for your project to be finalized and assigned to a team, please:
> 1. **Review all sections below** and update or expand any content as needed, making sure to address the SME Feedback in the section immediately below. Look for square brackets to find the places below that require additional inputs from you (e.g., "About [Company / Org Name]").
> 2. **Add your dataset** to the [data folder](data) in this repo.
> 3. **Close the Issue assigned to you in this repo** to let us know that you have made your edits and the overview page is ready for final review. You can do this by going to the _Issues_ tab in the top left section of the menu above, add a comment that says "CA review complete", and click the button to Close the Issue. 
>
> If you're unfamiliar with how to edit a page like this in GitHub, check out [this tutorial](https://ubc-lib-geo.github.io/gis-workshop-waml-template/content/handson/edit-readme.html) for a quick overview (start with step 2 and only edit this page), and [this guide](https://ubc-lib-geo.github.io/gis-workshop-waml-template/content/markdown.html) on how to use Markdown to compose text.
>
>
> ❌ Remember that this is a public repo. Do NOT include: Proprietary data, PII, API keys, credentials, or anything confidential.

---
## 📋 BTT Internal Evaluation Notes

| Check                       | Status | Notes |
|-----------------------------|--------|-------|
| Python Compatibility         | 🟢  | The tech stack focuses on Python-compatible tools and libraries for NLP and RAG, ensuring alignment with students' skill sets. |
| Data Readiness               | 🟢  | The public financial documents are accessible and under 1GB, making them readily usable with minimal preparation. This aligns with the accessibility requirement for students. |
| Resource Check               | 🟢  | The project uses free-tier tools such as Google Colab, ensuring all students have access to the necessary resources without specialized hardware or software constraints. |

**Student Fit Score:** 8/10  
**Technical Depth Score:** 7/10  
**Overall Recommendation:** REVISE

**Advisor Feedback Draft:**
The proposal makes excellent use of existing public data, ensuring it taps into a real-world scenario. However, a clearer definition of the evaluation process is needed to ensure that the model's performance can be quantitatively assessed against industry benchmarks. Additionally, providing explicit examples of the types of financial questions the system is expected to handle would enhance clarity. I recommend focusing on these aspects as critical adjustments to bolster the project’s robustness.

---

# Extracting Insight from Financial Documents Using RAG

**Company / Org:** 7-11  
**Challenge Advisor:** SaiSandeep Kantareddy   
**AI Coach:** Srihari Kamath, srihari.kamath@breakthroughtech.org   
**Program:** Break Through Tech AI Studio - Fall 2026

----

## 🏢 About 7-11

7-11 is a leading convenience store chain specializing in providing a wide variety of food, beverages, and daily essentials. Our focus is on enhancing customer experience while efficiently managing our retail operations.

----

## 🎯 The Challenge

### Project Summary
In this project, you will use public financial documents, including SEC filings and earnings call transcripts, and retrieval-augmented generation (RAG), semantic search, reranking, and automated evaluation techniques to build a system that answers financial questions with grounded citations from source documents. This will help our company address the business problem of extracting reliable insights from long, unstructured documents quickly and accurately.

### Success Criteria
Top-k retrieval accuracy, answer relevance, citation correctness, and groundedness; must outperform keyword-search baseline.

### Project Milestones

Use these milestones to guide your work. Your team will create a **GitHub Projects board** to track tasks within each milestone.

| Month | Milestone | Key Activities |
| :--- | :--- | :--- |
| September | Data Ingestion, Text Chunking & Embedding Indexing | • Ingest FinanceBench dataset and SEC 10-K/10-Q filings from EDGAR.<br>• Implement document parsing, financial table extraction, and semantic text chunking strategies.<br>• Generate vector embeddings and construct a vector database index (e.g., ChromaDB, FAISS, pgvector).<br>• Establish baseline evaluation metrics using naive vector search and vanilla LLM Q&A. |
| October | RAG Pipeline Development, Advanced Retrieval & Grounding | • Implement hybrid search combining dense vector retrieval and sparse keyword (BM25) search.<br>• Integrate re-ranking models (e.g., Cross-Encoder) and query transformation techniques.<br>• Enforce strict grounding prompt constraints and citation attribution to eliminate hallucinations in financial responses.<br>• Evaluate retrieval accuracy and answer correctness against FinanceBench benchmarks using RAGAS/TruLens metrics. |
| November / December | System Optimization, Interactive UI & Capstone Deliverables | • Optimize retrieval latency, token efficiency, and chunking over multi-page financial tables.<br>• Build an interactive Streamlit application enabling users to query financial reports, view exact cited source snippets, and inspect grounding scores.<br>• Finalize clean, reproducible GitHub repository, comprehensive technical documentation, and stakeholder presentation deck. |

### Stretch Goals
* **Multimodal Financial Table Parsing:** Integrate layout-aware parsing models (e.g., LlamaParse or Unstructured) to accurately parse complex nested financial tables, balance sheets, and footnote data.
* **Financial Metric Calculation Engine:** Build an automated numerical verification module that validates derived financial calculations (e.g., YoY growth rates, operating margins) against raw source table values.
* **Real-Time Grounding Guardrail Middleware:** Implement an automated interception layer that evaluates generated responses against retrieved context snippets and flags or blocks ungrounded claims before displaying output.
> **Note for the team:** Please create a GitHub Projects board in this repository to break these milestones into weekly tasks. Go to the **Projects** tab → **New project** → Choose **Board** → Add columns for each month.

---

## 📊 Dataset

**Name and Source:** Public financial documents (SEC filings and earnings call transcripts) from SEC EDGAR and FinanceBench dataset  
**Format:** Text and JSON  
**Size:** under 1gb  
**Location:** [Link to dataset or instructions for accessing it]

### Key Details
- Public financial documents (SEC filings and earnings call transcripts) from SEC EDGAR and FinanceBench dataset. Formats include Text and JSON.
- No known limitations; the dataset requires minimal preprocessing.
- [Link to data dictionary or documentation, if available]

---

## 🛠️ Suggested Approach

**ML Problem Type:** NLP

**Recommended Libraries:**
- [e.g., pandas, scikit-learn, TensorFlow, Hugging Face]

**Evaluation Metrics:**
- Accuracy, Precision/Recall, BLEU score

---

## 📚 Resources to Get Started

The following resources will help your team understand the problem space and potential technical approaches for this project:

**Background Reading:**
- [e.g., Link to an article or blog post about the problem domain]
- [e.g., Link to an industry report or case study]

**Technical Tutorials:**
- [e.g., Link to a free tutorial on the ML technique(s) involved]
- [e.g., Link to documentation for a key library or tool]

**Code Examples:**
- [e.g., Link to a relevant GitHub repo]
- [e.g., Link to a sample implementation or starter code]

**Other:**
- [Links to any additional resources — e.g., papers, videos, podcasts, etc.]

*Feel free to explore beyond these, and share anything interesting you find with me!*

---

## 🤝 How We'll Work Together

**Official check-ins:** During our biweekly 45-minute AI Studio Lab Section meeting block (2nd and 4th week of every month)

 **Other ways to reach out to me with questions:** 
* [e.g., Your team's channel within Break Through Tech’s Discord space]
* [e.g., Email; please copy your teammates and AI Studio Coach]
* [e.g., Request a team check-in on Zoom]
* [Note: I will aim to respond within 48 hours. Please reach out to your AI Studio Coach with urgent questions.]

> 💡 **Challenge Advisor: Please update the above based on your availability and preference. If you are not able to answer questions or meet with fellows outside of the biweekly Lab Section check-ins, simply write in "N/A (only available during the official check-in times)"**

**Recommended free coding / collaboration tools**
* […]
* […]

---

## 🚀 Getting Started

1. **Review this overview document** and note any questions for our first meeting
2. **Begin reviewing the dataset** using the link above
3. **Read the GitHub Projects documentation** [here](https://docs.github.com/en/issues/planning-and-tracking-with-projects/learning-about-projects/about-projects)

I’m excited to work with you!

---

## ❓ Questions?

Please bring any questions to our first meeting during the week of August 24th (Break Through Tech’s Bridge to Studio - Session C). 
