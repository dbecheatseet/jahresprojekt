import streamlit as st
import pandas as pd
from esco_kg_matching import ESCOKnowledgeGraph
from helpers import remove_emojis
import uuid
import requests
from difflib import SequenceMatcher
import numpy as np

# Cache für die Knowledge Graph Instanz
@st.cache_resource
def get_kg():
    kg = ESCOKnowledgeGraph(data_dir='data')
    kg.load_esco_data()
    kg._load_or_compute_embeddings()
    return kg

# Cache für die Mitarbeiterliste
@st.cache_data
def get_employee_list(_kg):
    return _kg.employees['employee_id'].tolist()

# Cache für die Berufsliste
@st.cache_data
def get_occupation_list(_kg):
    uris = [_kg.occupations.index[i] for i in range(len(_kg.occupations))]
    labels = {uri: _kg.occupations.loc[uri, 'preferredLabel'] for uri in uris}
    return uris, labels

# Cache für die Skills-Liste
@st.cache_data
def get_skills_list(_kg):
    # Verwende die Skills direkt aus dem Knowledge Graph
    skills = []
    for uri in _kg.skills.index:
        label = _kg.skills.loc[uri, 'preferredLabel']
        if isinstance(label, str):  # Stelle sicher, dass das Label ein String ist
            skills.append((uri, label))
    return sorted(skills, key=lambda x: x[1])  # Sortiere nach Label

# Cache für fehlende Skills
@st.cache_data
def get_missing_skills_cached(_kg, employee_id, target_occupation):
    return _kg.get_missing_skills(employee_id, target_occupation)

# Funktion zum Abrufen von Kursen von Coursera
@st.cache_data(ttl=3600)  # Cache für 1 Stunde
def get_coursera_courses(skill, limit=5):
    url = "https://api.coursera.org/api/courses.v1"
    params = {
        "q": "search",
        "query": skill,
        "limit": limit
    }
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Fehler beim Abrufen der Kurse: {str(e)}")
        return None

# Funktion zum Berechnen des Skill-Matchings
def calculate_skill_match(skill1, skill2):
    # Berechne die Ähnlichkeit zwischen zwei Skills
    return SequenceMatcher(None, skill1.lower(), skill2.lower()).ratio()

# Funktion zum Berechnen des Matching-Scores für einen Kurs
def calculate_course_match_score(course, missing_skills):
    if not course or 'elements' not in course:
        return 0
    
    total_score = 0
    matched_skills = []
    
    for element in course['elements']:
        course_name = element.get('name', '').lower()
        course_description = element.get('description', '').lower()
        
        for skill in missing_skills:
            skill_label = skill['skill_label'].lower()
            # Berechne Match-Score für den Skill
            name_match = calculate_skill_match(skill_label, course_name)
            desc_match = calculate_skill_match(skill_label, course_description)
            
            # Gewichte die Matches (Name ist wichtiger als Beschreibung)
            skill_score = (name_match * 0.7) + (desc_match * 0.3)
            
            if skill_score > 0.3:  # Mindestschwelle für einen Match
                total_score += skill_score * (2 if skill['occupation_skill_level'] == 'essential' else 1)
                matched_skills.append({
                    'skill': skill_label,
                    'score': skill_score,
                    'level': skill['occupation_skill_level']
                })
    
    return {
        'total_score': total_score,
        'matched_skills': matched_skills
    }

# Cache für die Kursempfehlungen
@st.cache_data(ttl=3600)  # Cache für 1 Stunde
def get_course_recommendations_cached(_kg, employee_id, target_occupation, top_k=3):
    # Hole fehlende Skills
    missing_skills = _kg.get_missing_skills(employee_id, target_occupation)
    if not missing_skills:
        return []
    
    # Hole Kurse für jeden fehlenden Skill
    all_courses = []
    for skill in missing_skills:
        courses = get_coursera_courses(skill['skill_label'])
        if courses and 'elements' in courses:
            for course in courses['elements']:
                match_score = calculate_course_match_score({'elements': [course]}, missing_skills)
                if match_score['total_score'] > 0:
                    all_courses.append({
                        'course': course,
                        'match_score': match_score,
                        'target_skill_level': skill['occupation_skill_level'],
                        'suitable_for_experience': skill.get('experience_level', 'Beginner')
                    })
    
    # Sortiere Kurse nach Match-Score
    all_courses.sort(key=lambda x: x['match_score']['total_score'], reverse=True)
    
    return all_courses[:top_k]

# Funktion zum Speichern eines neuen Mitarbeiters
def save_new_employee(kg, name, skills):
    # Generiere eine eindeutige ID
    employee_id = f"EMP{uuid.uuid4().hex[:8].upper()}"
    
    # Erstelle neuen Mitarbeiter
    new_employee = pd.DataFrame({
        'employee_id': [employee_id],
        'name': [name],
        'skills': [';'.join(skills)],
        'years_of_experience': [0],  # Standardwert
        'experience_level': ['Beginner']  # Standardwert
    })
    
    # Füge den neuen Mitarbeiter zur employees.csv hinzu
    kg.employees = pd.concat([kg.employees, new_employee], ignore_index=True)
    kg.employees.to_csv('data/employees.csv', index=False)
    
    return employee_id

# Funktion zum Löschen eines Mitarbeiters
def delete_employee(kg, employee_id):
    # Entferne den Mitarbeiter aus dem DataFrame
    kg.employees = kg.employees[kg.employees['employee_id'] != employee_id]
    # Speichere die Änderungen
    kg.employees.to_csv('data/employees.csv', index=False)
    # Lösche den Mitarbeiter aus dem Session State, falls er ausgewählt war
    if st.session_state.selected_employee == employee_id:
        st.session_state.selected_employee = None
        st.session_state.selected_occ = None

# Initialisierung
kg = get_kg()
employee_ids = get_employee_list(kg)
occ_uris, occ_labels = get_occupation_list(kg)
skills_list = get_skills_list(kg)

# Initialisiere Session State
if 'selected_employee' not in st.session_state:
    st.session_state.selected_employee = None
if 'selected_occ' not in st.session_state:
    st.session_state.selected_occ = None

# Tabs
tab1, tab2, tab3 = st.tabs(["👥 Mitarbeiter", "📋 Profil", "🎓 Kurse"])

# Tab 1: Mitarbeiterverwaltung
with tab1:
    st.header("Mitarbeiterverwaltung")
    
    # Auswahl zwischen bestehendem und neuem Mitarbeiter
    action = st.radio(
        "Was möchten Sie tun?",
        ["Bestehenden Mitarbeiter auswählen", "Neuen Mitarbeiter hinzufügen", "Mitarbeiter löschen"]
    )
    
    if action == "Bestehenden Mitarbeiter auswählen":
        selected_employee = st.selectbox(
            "Mitarbeiter auswählen",
            employee_ids,
            format_func=lambda x: kg.employees[kg.employees['employee_id'] == x]['name'].iloc[0]
        )
        
        # Speichere die Auswahl im Session State
        st.session_state.selected_employee = selected_employee
        
        # Zeige Mitarbeiterdetails
        emp = kg.employees[kg.employees['employee_id'] == selected_employee].iloc[0]
        st.subheader(f"Mitarbeiterdetails: {emp['name']}")
        
        # Aktuelle Skills anzeigen
        st.write("Aktuelle Skills:")
        if pd.notna(emp.get('skills')):
            for uri in emp['skills'].split(';'):
                if uri.strip() in kg.skills.index:
                    st.write(f"- {kg.skills.loc[uri.strip(), 'preferredLabel']}")
    
    elif action == "Neuen Mitarbeiter hinzufügen":
        st.subheader("Neuen Mitarbeiter hinzufügen")
        
        # Eingabefelder
        name = st.text_input("Name des Mitarbeiters")
        
        # Skills-Auswahl
        st.write("Skills auswählen:")
        selected_skills = st.multiselect(
            "Skills",
            options=[s[0] for s in skills_list],  # URIs
            format_func=lambda x: next((s[1] for s in skills_list if s[0] == x), x)  # Labels
        )
        
        if st.button("Mitarbeiter speichern"):
            if name and selected_skills:
                try:
                    employee_id = save_new_employee(kg, name, selected_skills)
                    st.success(f"Mitarbeiter {name} wurde erfolgreich hinzugefügt!")
                    # Aktualisiere die Mitarbeiterliste
                    st.cache_data.clear()
                    employee_ids = get_employee_list(kg)
                    # Setze den neuen Mitarbeiter als ausgewählt
                    st.session_state.selected_employee = employee_id
                except Exception as e:
                    st.error(f"Fehler beim Speichern: {str(e)}")
            else:
                st.warning("Bitte füllen Sie alle Felder aus.")
    
    else:  # Mitarbeiter löschen
        st.subheader("Mitarbeiter löschen")
        
        # Auswahl des zu löschenden Mitarbeiters
        employee_to_delete = st.selectbox(
            "Zu löschenden Mitarbeiter auswählen",
            employee_ids,
            format_func=lambda x: kg.employees[kg.employees['employee_id'] == x]['name'].iloc[0]
        )
        
        # Bestätigungsdialog
        if st.button("Mitarbeiter löschen", type="primary"):
            try:
                employee_name = kg.employees[kg.employees['employee_id'] == employee_to_delete]['name'].iloc[0]
                delete_employee(kg, employee_to_delete)
                st.success(f"Mitarbeiter {employee_name} wurde erfolgreich gelöscht!")
                # Aktualisiere die Mitarbeiterliste
                st.cache_data.clear()
                employee_ids = get_employee_list(kg)
            except Exception as e:
                st.error(f"Fehler beim Löschen: {str(e)}")

# Tab 2: Profil
with tab2:
    if st.session_state.selected_employee:
        emp = kg.employees[kg.employees['employee_id'] == st.session_state.selected_employee].iloc[0]
        st.header(f"Profil: {remove_emojis(emp['name'])}")

        # Aktuelle Skills
        st.subheader("Aktuelle Skills")
        if pd.notna(emp.get('skills')):
            for uri in emp['skills'].split(';'):
                data = kg.skills.loc[uri.strip()] if uri.strip() in kg.skills.index else None
                if data is not None:
                    label = remove_emojis(data['preferredLabel'])
                    st.markdown(f"- **{label}**")
        else:
            st.markdown("Keine Skills erfasst.")

        # Gewählte Zielrolle
        st.subheader("Gewählte Zielrolle")
        selected_occ = st.selectbox(
            "Zielrolle auswählen",
            occ_uris,
            format_func=lambda u: remove_emojis(occ_labels.get(u, u))
        )
        # Speichere die Auswahl im Session State
        st.session_state.selected_occ = selected_occ
        st.markdown(f"**{remove_emojis(occ_labels.get(selected_occ, selected_occ))}**")

        # Fehlende Skills
        st.subheader("Fehlende Skills")
        missing = get_missing_skills_cached(kg, st.session_state.selected_employee, selected_occ)
        if missing:
            # Gruppiere Skills nach Level
            essential_skills = [m for m in missing if m['occupation_skill_level'] == 'essential']
            optional_skills = [m for m in missing if m['occupation_skill_level'] == 'optional']
            
            if essential_skills:
                st.markdown("**Essentielle Skills:**")
                for m in essential_skills:
                    st.markdown(f"- **{remove_emojis(m['skill_label'])}**")
            
            if optional_skills:
                st.markdown("**Optionale Skills:**")
                for m in optional_skills:
                    st.markdown(f"- **{remove_emojis(m['skill_label'])}**")
        else:
            st.markdown("Keine fehlenden Skills für die gewählte Rolle.")
    else:
        st.info("Bitte wählen Sie zuerst einen Mitarbeiter aus dem Tab 'Mitarbeiter' aus.")

# Tab 3: Kurse
with tab3:
    st.header("Kursempfehlungen")
    if st.session_state.selected_employee and st.session_state.selected_occ:
        recs = get_course_recommendations_cached(kg, st.session_state.selected_employee, st.session_state.selected_occ, top_k=3)
        if recs:
            for i, rec in enumerate(recs, 1):
                c = rec['course']
                with st.expander(f"📚 {remove_emojis(c['name'])} (Match-Score: {rec['match_score']['total_score']:.2f})"):
                    st.write(c.get('description', '_Keine Beschreibung_'))
                    st.write(f"**Skill-Level:** {rec['target_skill_level']}")
                    st.write(f"**Erfahrung:** {rec['suitable_for_experience']}")
                    
                    # Zeige gematchte Skills
                    st.write("**Gematchte Skills:**")
                    for skill_match in rec['match_score']['matched_skills']:
                        st.write(f"- {skill_match['skill']} (Score: {skill_match['score']:.2f}, Level: {skill_match['level']})")
                    
                    # Link zum Kurs
                    if 'slug' in c:
                        st.markdown(f"[Zum Kurs auf Coursera](https://www.coursera.org/learn/{c['slug']})")
        else:
            st.info("Keine passenden Kursempfehlungen gefunden. Versuchen Sie es mit einer anderen Zielrolle.")
    else:
        st.info("Bitte wählen Sie zuerst einen Mitarbeiter und eine Zielrolle aus.")