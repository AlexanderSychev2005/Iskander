import json
import os

# --- MAPPING DICTIONARIES ---

def map_period(p):
    if not p: return 'Unknown'
    p = p.lower()
    if 'neo-assyrian' in p and not 'neo/late' in p and not 'archaic' in p: return 'Neo-Assyrian'
    if 'old babylonian' in p: return 'Old Babylonian'
    if 'middle babylonian' in p: return 'Middle Babylonian'
    if 'middle assyrian' in p: return 'Middle Assyrian'
    if 'neo-babylonian' in p or 'late babylonian' in p or 'neo/late babylonian' in p: return 'Neo/Late Babylonian'
    if p in ['ed iiia', 'ur iii', 'ebla', 'old akkadian', 'ed iiib', 'early dynastic', 'uruk iii', 'uruk iv', 'archaic', 'lagaš ii', 'fara']: return 'Third Millennium'
    if p in ['seleucid', 'achaemenid', 'hellenistic', 'persian']: return 'Late Antiquity'
    if 'unknown' in p or 'uncertain' in p or 'unclear' in p: return 'Unknown'
    return 'Other'

def map_provenience(p):
    if not p: return 'Unknown'
    p = p.lower()
    if 'nineveh' in p or 'kuyunjik' in p or 'ninua' in p: return 'Nineveh'
    if 'aššur' in p or 'assur' in p or 'qalʿat sherqat' in p or 'ashur' in p: return 'Assur'
    if 'nippur' in p or 'nuffar' in p: return 'Nippur'
    if 'uruk' in p or 'warka' in p: return 'Uruk'
    if 'nimrud' in p or 'kalhu' in p: return 'Nimrud'
    if 'ugarit' in p or 'emar' in p or 'hattusa' in p or 'alalakh' in p or 'byblos' in p or 'ḫattusa' in p: return 'Levant & Anatolia'
    if 'unclear' in p or 'unknown' in p or 'uncertain' in p: return 'Unknown'
    return 'Other' # Will catch Babylon, Sippar, etc.

def map_genre(g):
    if not g: return 'Unknown'
    g = g.lower()
    if 'royal' in g or 'monumental' in g:
        if 'ritual' in g: return 'Literary & Ritual'
        return 'Royal Inscription'
    if 'lexical' in g or 'scholarly' in g or 'astrological' in g or 'omen' in g or 'extispicy' in g or 'medical' in g or 'mathematical' in g or 'astronomical' in g: return 'Lexical & Scholarly'
    if 'administrative' in g or 'eponym list' in g or 'appointment' in g: return 'Administrative'
    if 'literary' in g or 'ritual' in g or 'hymn' in g or 'prophecy' in g or 'epic' in g: return 'Literary & Ritual'
    if 'legal' in g or 'treaty' in g or 'grant' in g or 'gift' in g: return 'Legal'
    if 'letter' in g: return 'Letter'
    if 'unknown' in g: return 'Unknown'
    return 'Other'

def map_language(l):
    if not l: return 'Unknown'
    l = l.lower()
    if l == 'akkadian' or l == 'middle assyrian' or l == 'old assyrian' or l == 'standard babylonian' or 'akkadian (with' in l: return 'Akkadian'
    if l == 'sumerian' or l == 'sumerian ?': return 'Sumerian'
    if 'bilingual' in l or ('sumerian' in l and 'akkadian' in l): return 'Bilingual'
    if 'urartian' in l or 'hittite' in l or 'eblaite' in l or 'elamite' in l or 'old persian' in l: return 'Other Ancient'
    if 'unknown' in l or 'undetermined' in l: return 'Unknown'
    return 'Other'

def map_ruler(r):
    if not r: return 'Unknown'
    r_lower = r.lower()
    if r_lower == 'ashurbanipal': return 'Ashurbanipal'
    if r_lower == 'esarhaddon': return 'Esarhaddon'
    if r_lower == 'sargon ii': return 'Sargon II'
    if r_lower == 'sennacherib': return 'Sennacherib'
    if r_lower == 'esarhaddon or ashurbanipal': return 'Esarhaddon or Ashurbanipal'
    if r_lower == 'nabonidus': return 'Nabonidus'
    if r_lower == 'nebuchadnezzar ii': return 'Nebuchadnezzar II'
    if r_lower == 'tiglath-pileser iii': return 'Tiglath-pileser III'
    if r_lower == 'ashurnasirpal ii': return 'Ashurnasirpal II'
    if 'unknown' in r_lower or 'uncertain' in r_lower: return 'Unknown'
    return 'Other Ruler'

def main():
    input_path = r'C:\Programming\akkadian\data\oracc_dataset\oracc_unique.jsonl'
    output_path = r'C:\Programming\akkadian\data\oracc_dataset\oracc_processed.jsonl'
    
    print("Preprocessing labels and text gaps...")
    
    with open(input_path, 'r', encoding='utf-8') as fin, open(output_path, 'w', encoding='utf-8') as fout:
        for line in fin:
            try:
                data = json.loads(line)
            except:
                continue
                
            # 1. Unify text gaps: replace all uppercase 'X' with 'x' in signs
            if 'signs' in data and data['signs']:
                if isinstance(data['signs'], list):
                    data['signs'] = "".join(data['signs'])
                data['signs'] = data['signs'].replace('X', 'x')
                
            # 2. Map metadata
            data['period_mapped'] = map_period(data.get('period', ''))
            data['provenience_mapped'] = map_provenience(data.get('provenience', ''))
            data['genre_mapped'] = map_genre(data.get('genre', ''))
            data['language_mapped'] = map_language(data.get('language', ''))
            data['ruler_mapped'] = map_ruler(data.get('ruler', ''))
            
            # Write to new file
            fout.write(json.dumps(data, ensure_ascii=False) + '\n')
            
    print(f"Processed dataset saved to {output_path}")

if __name__ == '__main__':
    main()
