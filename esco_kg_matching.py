import os
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import joblib
from functools import lru_cache

class ESCOKnowledgeGraph:
    def __init__(self, data_dir='data'):
        self.data_dir = data_dir
        self.skills = None
        self.occupations = None
        self.occupation_skills = None
        self.courses = None
        self.course_skill_rel = None
        self.employees = None
        self.model = None
        self.skill_embeddings = None
        self.course_embeddings = None
        
        # Cache-Dateien
        self.embeddings_cache_file = os.path.join(data_dir, 'embeddings_cache.joblib')
        
        # Lazy Loading der Daten
        self._load_employees()

    def _load_employees(self):
        """Lazy Loading der Mitarbeiterdaten"""
        if self.employees is None:
            self.employees = pd.read_csv(
                os.path.join(self.data_dir, "employees.csv"), encoding='utf-8'
            )

    def load_esco_data(self):
        """Lädt ESCO-Daten mit optimierter Speichernutzung"""
        # Skills
        self.skills = pd.read_csv(
            os.path.join(self.data_dir, "skills_de.csv"), encoding='utf-8'
        ).set_index('conceptUri')
        
        # Occupations
        self.occupations = pd.read_csv(
            os.path.join(self.data_dir, "occupations_de.csv"), encoding='utf-8'
        ).set_index('conceptUri')
        
        # Occupation-Skill Relations
        self.occupation_skills = pd.read_csv(
            os.path.join(self.data_dir, "occupationSkillRelations_de.csv"), encoding='utf-8'
        )
        
        # Kurse
        self.courses = pd.read_csv(
            os.path.join(self.data_dir, "courses.csv"), encoding='utf-8'
        ).set_index('course_id')
        
        # Kurs→Skill-Mapping
        self.course_skill_rel = pd.read_csv(
            os.path.join(self.data_dir, "courseSkillRelations.csv"), encoding='utf-8'
        )

    def _load_or_compute_embeddings(self):
        """Lädt oder berechnet Embeddings mit Caching"""
        if os.path.exists(self.embeddings_cache_file):
            cached_data = joblib.load(self.embeddings_cache_file)
            self.skill_embeddings = cached_data['skill_embeddings']
            self.course_embeddings = cached_data['course_embeddings']
        else:
            self.compute_embeddings()
            # Cache speichern
            joblib.dump({
                'skill_embeddings': self.skill_embeddings,
                'course_embeddings': self.course_embeddings
            }, self.embeddings_cache_file)

    def compute_embeddings(self):
        """Berechnet Embeddings mit optimierter Batch-Verarbeitung"""
        if self.skills is None or self.courses is None:
            raise ValueError("Bitte zuerst load_esco_data() aufrufen.")

        if self.model is None:
            self.model = SentenceTransformer('all-mpnet-base-v2')

        # Batch-Verarbeitung für Skills
        skill_texts = self.skills['preferredLabel'].fillna('').tolist()
        self.skill_embeddings = self.model.encode(
            skill_texts, 
            show_progress_bar=True,
            batch_size=32
        )

        # Batch-Verarbeitung für Kurse
        course_texts = self.courses['course_name'].fillna('').tolist()
        self.course_embeddings = self.model.encode(
            course_texts,
            show_progress_bar=True,
            batch_size=32
        )

    @lru_cache(maxsize=1000)
    def get_missing_skills(self, employee_id, target_occupation):
        """Optimierte Version mit Caching für wiederholte Aufrufe"""
        emp = self.employees[self.employees['employee_id'] == employee_id]
        if emp.empty:
            return []
            
        exp = emp.iloc[0]['years_of_experience']
        raw_skills = emp.iloc[0].get('skills', '')
        have = set(raw_skills.split(';')) if pd.notna(raw_skills) else set()

        req = self.occupation_skills[
            self.occupation_skills['occupationUri'] == target_occupation
        ]
        
        missing = []
        for _, r in req.iterrows():
            uri = r['skillUri']
            if uri not in have:
                label = self.skills.loc[uri, 'preferredLabel'] if uri in self.skills.index else uri
                level = r.get('relationType', 'N/A')
                missing.append({
                    'skill_uri': uri,
                    'skill_label': label,
                    'occupation_skill_level': level,
                    'skill_level': level,
                    'employee_experience': exp
                })
        return missing

    @lru_cache(maxsize=100)
    def recommend_courses(self, employee_id, target_occupation, top_k=3):
        """Optimierte Version mit Caching für wiederholte Aufrufe"""
        missing = self.get_missing_skills(employee_id, target_occupation)
        if not missing:
            return []

        # Missing URIs
        uris = [m['skill_uri'] for m in missing]
        
        # Optimierte Filterung
        mapped = self.course_skill_rel[
            self.course_skill_rel['skillUri'].isin(uris)
        ]
        if mapped.empty:
            return []

        # Einzigartige Kandidaten
        candidate_ids = mapped['course_id'].unique()
        candidates = self.courses.loc[candidate_ids]
        
        if candidates.empty:
            return []

        # Embeddings berechnen oder laden
        if self.skill_embeddings is None or self.course_embeddings is None:
            self._load_or_compute_embeddings()

        # Optimierte Similarity-Berechnung
        skill_idxs = [
            self.skills.index.get_loc(uri)
            for uri in uris if uri in self.skills.index
        ]
        course_idxs = [
            self.courses.index.get_loc(cid)
            for cid in candidate_ids
        ]

        skill_emb = self.skill_embeddings[skill_idxs]
        cand_emb = self.course_embeddings[course_idxs]
        
        sims = cosine_similarity(skill_emb, cand_emb)
        top = sims.mean(axis=0).argsort()[-top_k:][::-1]

        results = []
        for i in top:
            c = candidates.iloc[i]
            results.append({
                'course': {
                    'course_id': c.name,
                    'course_name': c['course_name'],
                    'description': c.get('description', '')
                },
                'target_skill_level': missing[0]['skill_level'],
                'suitable_for_experience': missing[0]['employee_experience']
            })
        return results

if __name__ == "__main__":
    kg = ESCOKnowledgeGraph(data_dir='data')
    kg.load_esco_data()
    kg._load_or_compute_embeddings()
    print(kg.recommend_courses("EMP001", "http://data.europa.eu/esco/occupation/12345"))
