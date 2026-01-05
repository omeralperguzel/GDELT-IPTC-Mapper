#!/usr/bin/env python3
"""
GDELT Theme -> IPTC Media Topics Mapper (v3)
Hierarchical Approach: Top-level + Subtopic Definitions

Layer 1: Rule-based mapping using theme prefixes + Vargo issue taxonomy
Layer 2: Semantic embedding with Sentence-BERT + cosine similarity
Layer 3: Subtopic refinement with enhanced similarity scoring
Fusion: Hierarchical decision tree (top-level → subtopic)

Author: Omer Alper Guzel
Date: January 2026
"""

import pandas as pd
import numpy as np
import json
import re
from pathlib import Path
from datetime import datetime

# Sentence-Transformers
try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.manifold import TSNE
    from sklearn.decomposition import PCA
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("[!] sentence-transformers not installed. Run: pip install sentence-transformers")

# ============================================================================
# IPTC MEDIA TOPICS - 17 TOP-LEVEL CATEGORIES + SUBTOPICS (2025-10-10 Standard)
# ============================================================================

IPTC_CATEGORIES_V3 = [
    {
        "id": "01000000",
        "label": "arts, culture, entertainment and media",
        "definition": "All forms of arts, entertainment, cultural heritage and media",
        "subtopics": [
            {"id": "20000002", "label": "arts and entertainment", "definition": "All forms of arts and entertainment"},
            {"id": "20000038", "label": "culture", "definition": "The ideas, customs, arts, traditions of a particular group of persons"},
            {"id": "20000045", "label": "mass media", "definition": "Media addressing a large audience"}
        ]
    },
    {
        "id": "02000000",
        "label": "crime, law and justice",
        "definition": "Crime, legal proceedings, police, courts, punishment",
        "subtopics": [
            {"id": "20000082", "label": "crime", "definition": "Violations of laws by individuals, companies or organisations"},
            {"id": "20000106", "label": "judiciary", "definition": "The system of courts of law"},
            {"id": "20000121", "label": "law", "definition": "The codification of rules of behaviour"},
            {"id": "20000129", "label": "law enforcement", "definition": "Agencies that attempt to prevent disobedience to established laws or bring to justice those who disobey those laws"}
        ]
    },
    {
        "id": "03000000",
        "label": "disaster, accident and emergency incident",
        "definition": "Man made or natural event resulting in loss of life or injury to living creatures and/or damage to inanimate objects or property",
        "subtopics": [
            {"id": "20000139", "label": "accident and emergency incident", "definition": "A sudden, unexpected event that causes unwanted consequences or requires immediate action"},
            {"id": "20000148", "label": "disaster", "definition": "A sudden, unplanned event that causes great damage or loss of life, such as an accident or a natural catastrophe"},
            {"id": "20000168", "label": "emergency response", "definition": "The planning and efforts made by people and organizations to help victims of a sudden, unplanned event, natural disaster or crisis"}
        ]
    },
    {
        "id": "04000000",
        "label": "economy, business and finance",
        "definition": "All matters concerning the planning, production and exchange of wealth.",
        "subtopics": [
            {"id": "20000349", "label": "business enterprise", "definition": "Organisations set up to create and sell a product or service"},
            {"id": "20000170", "label": "business information", "definition": "Information about individual business entities, including companies, corporations, charities"},
            {"id": "20000344", "label": "economy", "definition": "Production, consumption, distribution and trade activities affecting regions or countries as a whole"},
            {"id": "20000385", "label": "market and exchange", "definition": "Market for buying and selling stocks, currencies, commodities and other goods"},
            {"id": "20000209", "label": "products and services", "definition": "Products and services consumed by companies and individuals and the companies that manufacture or manage them"}
        ]
    },
    {
        "id": "05000000",
        "label": "education",
        "definition": "All aspects of furthering knowledge, formally or informally",
        "subtopics": [
            {"id": "20000412", "label": "curriculum", "definition": "The courses offered by a learning institution and the regulation of those courses"},
            {"id": "20001217", "label": "educational grading", "definition": "The evaluation of a student's achievement on a test, assignment or course and the policies and methods around assigning those grades"},
            {"id": "20000413", "label": "educational testing and examinations", "definition": "Polices and standards around the testing and assessment of students, including the merits of standardised testing, and testing methods"},
            {"id": "20000414", "label": "entrance examination", "definition": "Exams for entering colleges, universities and all other higher and lower education institutions"},
            {"id": "20001337", "label": "online and remote learning", "definition": "Learning where a student and teacher are not physically present in the same location"},
            {"id": "20000398", "label": "parents group", "definition": "Group of parents set up to support educational activities of their children"},
            {"id": "20000399", "label": "religious education", "definition": "Instruction by any faith about that faith's principles and beliefs"},
            {"id": "20000400", "label": "school", "definition": "A building or institution in which education is provided"},
            {"id": "20000410", "label": "social learning", "definition": "The learning of social skills and behaviours through the imitation and observation of others"},
            {"id": "20000415", "label": "students", "definition": "Students as a demographic, including student protests and trends"},
            {"id": "20000416", "label": "teachers", "definition": "Teachers as a demographic, including teacher unions, teacher education and training"},
            {"id": "20001216", "label": "vocational education", "definition": "Education that provides students with practical experience and training in a particular occupational field"}
        ]
    },
    {
        "id": "06000000",
        "label": "environment",
        "definition": "The protection, damage, and condition of the ecosystem of the planet Earth and its surroundings",
        "subtopics": [
            {"id": "20000418", "label": "climate change", "definition": "All issues relating to extreme changes in climate, including rising global temperature and greenhouse gases"},
            {"id": "20000420", "label": "conservation", "definition": "Preservation of the natural world, such as wilderness areas, flora and fauna"},
            {"id": "20000424", "label": "environmental pollution", "definition": "The contamination of natural resources by harmful substances"},
            {"id": "20000430", "label": "natural resource", "definition": "Assets afforded by nature without human intervention that can be used for various purposes"},
            {"id": "20000441", "label": "nature", "definition": "The natural world"},
            {"id": "20001374", "label": "sustainability", "definition": "Actions by organizations, governments, and individuals in response to environmental problems"}
        ]
    },
    {
        "id": "07000000",
        "label": "health",
        "definition": "All aspects of physical and mental well-being",
        "subtopics": [
            {"id": "20000446", "label": "disease and condition", "definition": "Any health conditions affecting humans"},
            {"id": "20000480", "label": "government health care", "definition": "Health care provided by governments at any level"},
            {"id": "20000461", "label": "health facility", "definition": "Facilities used for any kind of health care"},
            {"id": "20000483", "label": "health insurance", "definition": "Insurance covering medical costs"},
            {"id": "20000463", "label": "health organisation", "definition": "Specific health organisations, including professional associations, non-profit and international groups"},
            {"id": "20000464", "label": "health treatment and procedure", "definition": "Remedies, therapies, interventions, medications, testing and treatments"},
            {"id": "20000485", "label": "medical profession", "definition": "Profession requiring formal training in study, diagnosis, treatment and prevention"},
            {"id": "20000484", "label": "private health care", "definition": "Health care provided by private organisations"},
            {"id": "20001358", "label": "public health", "definition": "Organised measures to prevent disease and promote health across populations"}
        ]
    },
    {
        "id": "08000000",
        "label": "human interest",
        "definition": "Item that discusses individuals, groups, animals, plants or other objects in an emotional way",
        "subtopics": [
            {"id": "20001237", "label": "anniversary", "definition": "The celebration or commemoration of a significant amount of years since an event or a notable person's birth or death"},
            {"id": "20000498", "label": "award and prize", "definition": "The recognition of an achievement in the form of a symbolic item or monetary gift"},
            {"id": "20001238", "label": "birthday", "definition": "A celebration on the anniversary of a person's birth"},
            {"id": "20000505", "label": "celebrity", "definition": "Life and behaviour of famous people"},
            {"id": "20000501", "label": "ceremony", "definition": "Rituals, such as dedications or commemorations"},
            {"id": "20000504", "label": "high society", "definition": "Life and behaviour of the rich and socialites"},
            {"id": "20000503", "label": "human mishap", "definition": "Silly or stupid human errors"},
            {"id": "20000499", "label": "record and achievement", "definition": "Non-sport achievement by an individual or group that sets a new record"}
        ]
    },
    {
        "id": "09000000",
        "label": "labour",
        "definition": "Social aspects, organisations, rules and conditions affecting the employment of human effort",
        "subtopics": [
            {"id": "20000509", "label": "employment", "definition": "The state of having work, usually paid"},
            {"id": "20000521", "label": "employment legislation", "definition": "Laws governing employment"},
            {"id": "20000523", "label": "labour market", "definition": "The supply and demand of labour in an economy"},
            {"id": "20000524", "label": "labour relations", "definition": "The relationship between workers and employers"},
            {"id": "20000531", "label": "retirement", "definition": "The years after work"},
            {"id": "20000533", "label": "unemployment", "definition": "The state of being available to work but not having a job"},
            {"id": "20000536", "label": "unions", "definition": "Groups established to represent workers for better workplace conditions"}
        ]
    },
    {
        "id": "10000000",
        "label": "lifestyle and leisure",
        "definition": "Activities undertaken for pleasure, relaxation or recreation outside paid employment",
        "subtopics": [
            {"id": "20000538", "label": "leisure", "definition": "Activities carried out in one's spare time"},
            {"id": "20000565", "label": "lifestyle", "definition": "The way in which a person lives, including their style and possessions"},
            {"id": "20001339", "label": "wellness", "definition": "The active pursuit of good mental and physical health"}
        ]
    },
    {
        "id": "11000000",
        "label": "politics and government",
        "definition": "Local, regional, national and international exercise of power, the day-to-day running of government, and the relationships between governing bodies and states.",
        "subtopics": [
            {"id": "20000574", "label": "election", "definition": "The selection of government representatives by the casting of votes by the populace"},
            {"id": "20000593", "label": "government", "definition": "The systems, institutions and people who run a political entity"},
            {"id": "20000621", "label": "government policy", "definition": "An overall plan or course of action set out by a government intended to influence and guide decisions"},
            {"id": "20000638", "label": "international relations", "definition": "Relations between nations through negotiation, treaty, or diplomacy"},
            {"id": "20000646", "label": "non-governmental organisation (NGO)", "definition": "Groups officially outside of government that lobby, demonstrate or campaign on a wide range of issues"},
            {"id": "20000648", "label": "political prisoners and dissenters", "definition": "Individuals who put themselves at risk by expressing their political views and the imprisonment of individuals who speak out against a political authority"}
        ]
    },
    {
        "id": "12000000",
        "label": "religion",
        "definition": "Belief systems, institutions and people who provide moral guidance to followers",
        "subtopics": [
            {"id": "20000657", "label": "belief systems", "definition": "A set of beliefs prescribed by an institution or text often focusing on worship and moral guidelines"},
            {"id": "20000702", "label": "relations between religion and government", "definition": "Matters arising from the relationship between religions and a government"},
            {"id": "20000688", "label": "religious conflict", "definition": "Conflicts involving religious differences"},
            {"id": "20000697", "label": "religious facility", "definition": "Any facility where a group carries out its religious rites"},
            {"id": "20000690", "label": "religious festival and holiday", "definition": "Holy day or period of observance in a religion"},
            {"id": "20000703", "label": "religious leader", "definition": "Person(s) who have a ritual, juridical or leading role in their religion"},
            {"id": "20000696", "label": "religious ritual", "definition": "Established religious rituals such as mass, baptism or prayer meetings"},
            {"id": "20000705", "label": "religious text", "definition": "Texts regarded as holy or important by a religion"}
        ]
    },
    {
        "id": "13000000",
        "label": "science and technology",
        "definition": "All aspects pertaining to human understanding of, as well as methodical study and research of natural, formal and social sciences",
        "subtopics": [
            {"id": "20000710", "label": "biomedical science", "definition": "Application of biology-based science to medical fields"},
            {"id": "20000715", "label": "mathematics", "definition": "The study of structure, space, change and number"},
            {"id": "20000717", "label": "natural science", "definition": "Sciences that deal with matter, energy and the physical world"},
            {"id": "20000741", "label": "scientific institution", "definition": "Institution that carries out or governs scientific work"},
            {"id": "20000735", "label": "scientific research", "definition": "The scientific and methodical investigation to explain events or find solutions"},
            {"id": "20000755", "label": "scientific standards", "definition": "Established rules governing scientific and technological study"},
            {"id": "20000742", "label": "social sciences", "definition": "Study of human society such as anthropology, economics or sociology"},
            {"id": "20000756", "label": "technology and engineering", "definition": "Study and practice of industrial or applied sciences"}
        ]
    },
    {
        "id": "14000000",
        "label": "society",
        "definition": "The concerns, issues, affairs and institutions relevant to human social interactions, problems and welfare",
        "subtopics": [
            {"id": "20000768", "label": "communities", "definition": "A group of individuals actively sharing a common value or interest"},
            {"id": "20000788", "label": "demographic group", "definition": "A subset of society with shared traits"},
            {"id": "20000770", "label": "demographics", "definition": "The study of human populations and their characteristics"},
            {"id": "20000775", "label": "discrimination", "definition": "Unfair treatment of individuals or groups based on identity"},
            {"id": "20001373", "label": "diversity, equity and inclusion", "definition": "Efforts to promote fair treatment and participation"},
            {"id": "20000772", "label": "emigration", "definition": "Leaving one's country of residence to settle elsewhere"},
            {"id": "20000780", "label": "family", "definition": "A group of people related genetically or by legal bond"},
            {"id": "20000587", "label": "fundamental rights", "definition": "Basic political, social and economic rights usually upheld by law"},
            {"id": "20000771", "label": "immigration", "definition": "Movement of individuals to another country"},
            {"id": "20000799", "label": "social condition", "definition": "Circumstances affecting a person's life and welfare"},
            {"id": "20000802", "label": "social problem", "definition": "Issues related to human rights, welfare and societal concern"},
            {"id": "20000808", "label": "values", "definition": "Principles or standards of behaviour"},
            {"id": "20000817", "label": "welfare", "definition": "Help for those in need of food, housing, health and other services"}
        ]
    },
    {
        "id": "15000000",
        "label": "sport",
        "definition": "Competitive activity or skill that involves physical and/or mental effort and organisations and bodies involved in these activities",
        "subtopics": [
            {"id": "20000822", "label": "competition discipline", "definition": "Different types of sport which can be executed in competitions"},
            {"id": "20001103", "label": "disciplinary action in sport", "definition": "Actions, including fines and suspensions levied by sports organisations and teams"},
            {"id": "20001104", "label": "drug use in sport", "definition": "Drug use associated with sport activities, including doping, abuse, testing and permitted medical uses"},
            {"id": "20001301", "label": "sport achievement", "definition": "Records or honours earned by athletes for their performance"},
            {"id": "20001108", "label": "sport event", "definition": "An event featuring one or more sport competitions"},
            {"id": "20001124", "label": "sport industry", "definition": "Commercial issues related to sport"},
            {"id": "20001125", "label": "sport organisation", "definition": "Organisations or associations that govern sports"},
            {"id": "20001126", "label": "sport venue", "definition": "Gymnasiums, stadiums, arenas or facilities where sports events take place"},
            {"id": "20001323", "label": "sports coaching", "definition": "The staff responsible for the training and on-field management of a sports team"},
            {"id": "20001324", "label": "sports management and ownership", "definition": "The executive leadership and owners of a sports team"},
            {"id": "20001325", "label": "sports officiating", "definition": "Referees, umpires and other staff who enforce the rules of a sport"},
            {"id": "20001148", "label": "sports transaction", "definition": "The transfer, hiring or drafting of athletes"}
        ]
    },
    {
        "id": "16000000",
        "label": "conflict, war and peace",
        "definition": "Armed conflicts, wars, peace processes, military operations",
        "subtopics": [
            {"id": "16010000", "label": "armed conflict", "definition": "Battles, military operations, casualties"},
            {"id": "16020000", "label": "peace processes", "definition": "Negotiations, ceasefires, diplomacy"},
            {"id": "16030000", "label": "post-conflict", "definition": "Reconstruction, reconciliation, peacekeeping"}
        ]
    },
    {
        "id": "17000000",
        "label": "weather",
        "definition": "Weather conditions, forecasts, climate patterns",
        "subtopics": [
            {"id": "17010000", "label": "meteorology", "definition": "Weather forecasting, atmospheric science"},
            {"id": "17020000", "label": "extreme weather", "definition": "Storms, floods, heat waves, cold snaps"},
            {"id": "17030000", "label": "climate patterns", "definition": "Seasonal weather, long-term trends"}
        ]
    }
]

# Flatten for embedding
def get_all_iptc_items(categories):
    """Get flattened list of all IPTC items (top-level + subtopics)"""
    items = []
    for cat in categories:
        items.append({
            "id": cat["id"],
            "label": cat["label"],
            "definition": cat["definition"],
            "level": "top",
            "parent_id": None
        })
        for sub in cat.get("subtopics", []):
            items.append({
                "id": sub["id"],
                "label": sub["label"],
                "definition": sub["definition"],
                "level": "sub",
                "parent_id": cat["id"]
            })
    return items

IPTC_ALL_ITEMS = get_all_iptc_items(IPTC_CATEGORIES_V3)

# ============================================================================
# LAYER 1: RULE-BASED MAPPING (EXTENDED FOR SUBTOPICS)
# ============================================================================

# Rule patterns now map to subtopic IDs where possible
RULE_PATTERNS_V3 = {
    # Arts subtopics
    r'^(ART|MUSEUM|PAINTING|SCULPTURE)': '20000002',  # arts and entertainment (medtop:20000002)
    r'^(CULTURE|HERITAGE|TRADITION)': '20000038',     # culture (medtop:20000038)
    r'^(ENTERTAINMENT|CELEBRITY)': '20000002',        # entertainment -> arts and entertainment (medtop:20000002)
    r'^(MEDIA|PROPAGANDA|CENSOR)': '20000045',       # mass media (medtop:20000045)

    # Crime subtopics
    r'^(CRIME|KIDNAP|RAPE|DRUG|MURDER|ASSAULT)': '20000082',  # crime (medtop:20000082)
    r'^(ARREST|PRISON|POLICE)': '20000129',                   # law enforcement (medtop:20000129)
    r'^(COURT|TRIAL|VERDICT)': '20000106',                    # judiciary (medtop:20000106)

    # Disaster subtopics (map to IPTC medtop qcodes)
    r'^(EARTHQUAKE|TSUNAMI|HURRICANE|FLOOD)': '20000148',     # disaster (medtop:20000148)
    r'^(ACCIDENT|EXPLOSION|FIRE)': '20000139',                # accident and emergency incident (medtop:20000139)
    r'^(EMERGENCY|EVACUATION|RESCUE)': '20000168',            # emergency response (medtop:20000168)

    # Economy subtopics (map to IPTC medtop qcodes)
    r'^(BUSINESS|CORPORATE|ENTREPRENEUR|COMPANY|COMPANIES)': '20000349',  # business enterprise (medtop:20000349)
    r'^(BUSINESS_INFO|BUSINESS_INFORMATION|COMPANY_INFO|REGISTRATION)': '20000170',  # business information (medtop:20000170)
    r'^(MARKET|EXCHANGE|STOCK|CURRENCY|COMMODITY)': '20000385',  # market and exchange (medtop:20000385)
    r'^(PRODUCT|PRODUCTS|SERVICE|SERVICES)': '20000209',  # products and services (medtop:20000209)
    r'^(ECONOMY|TRADE|GDP|UNEMPLOYMENT|INFLATION)': '20000344',  # economy (medtop:20000344)

    # Education subtopics (map to IPTC medtop qcodes)
    r'^(CURRICULUM|COURSE|SYLLABUS)': '20000412',
    r'^(GRADING|GRADE|EDUCATIONAL_GRADING)': '20001217',
    r'^(TEST|EXAM|EXAMINATION|ASSESSMENT)': '20000413',
    r'^(ENTRANCE|ADMISSION|ENTRANCE_EXAM)': '20000414',
    r'^(ONLINE|REMOTE|E_LEARNING|DISTANCE_LEARNING)': '20001337',
    r'^(PARENTS|PARENTS_GROUP|PTA)': '20000398',
    r'^(RELIGIOUS_EDUCATION|SUNDAY_SCHOOL)': '20000399',
    r'^(SCHOOL|PRIMARY|SECONDARY|SCHOOLING)': '20000400',
    r'^(SOCIAL_LEARNING|SOCIAL_SKILLS)': '20000410',
    r'^(STUDENT|STUDENTS)': '20000415',
    r'^(TEACHER|TEACHERS)': '20000416',
    r'^(VOCATIONAL|TECHNICAL|CAREER)': '20001216',

    # Environment subtopics (map to IPTC medtop qcodes)
    r'^(CLIMATE|EMISSION|GLOBAL_WARMING)': '20000418',
    r'^(CONSERVATION|BIODIVERSITY|WILDLIFE)': '20000420',
    r'^(POLLUTION|CONTAMINATION|WASTE)': '20000424',
    r'^(NATURAL_RESOURCE|RESOURCE|FOREST|MINERAL)': '20000430',
    r'^(NATURE|NATURAL_WORLD)': '20000441',
    r'^(SUSTAINABILITY|SUSTAINABLE)': '20001374',

    # Health subtopics (map to IPTC medtop qcodes)
    r'^(DISEASE|CONDITION|ILLNESS|INFECTION)': '20000446',
    r'^(GOVERNMENT_HEALTH|GOV_HEALTH|GOVERNMENT_HEALTHCARE)': '20000480',
    r'^(HOSPITAL|HEALTHCARE|CLINIC)': '20000461',
    r'^(INSURANCE|HEALTH_INSURANCE)': '20000483',
    r'^(HEALTH_ORGANISATION|HEALTH_ORG|NGO_HEALTH)': '20000463',
    r'^(TREATMENT|THERAPY|VACCINE|DRUG|SURGERY)': '20000464',
    r'^(MEDICAL_PROFESSION|DOCTOR|NURSE|PROFESSION)': '20000485',
    r'^(PRIVATE_HEALTH|PRIVATE_HOSPITAL)': '20000484',
    r'^(PUBLIC_HEALTH|EPIDEMIC|PANDEMIC)': '20001358',

    # Human interest subtopics
    r'^(ANNIVERSARY|ANNIVERSARIES|ANNIV|ANNIVERSARY_EVENT|BIRTHDAY)': '20001237',
    r'^(AWARD|PRIZE|GRANT|HONOUR|HONOR)': '20000498',
    r'^(BIRTHDAY|BIRTHDAY_EVENT)': '20001238',
    r'^(CELEBRITY|FAMOUS|STAR)': '20000505',
    r'^(CEREMONY|RITUAL_EVENT)': '20000501',
    r'^(HIGH_SOCIETY|SOCIETY|SOCIETAL_ELITE)': '20000504',
    r'^(MISHAP|HUMAN_MISHAP|SILLY_EVENT)': '20000503',
    r'^(RECORD|ACHIEVEMENT|RECORDS)': '20000499',

    # Religion subtopics (map to IPTC medtop qcodes)
    r'^(CHURCH|MOSQUE|TEMPLE|RELIGIOUS_FACILITY)': '20000697',
    r'^(RITUAL|CEREMONY|RELIGIOUS_RITUAL)': '20000696',
    r'^(FAITH|THEOLOGY|RELIGIOUS_TEXT)': '20000705',
    r'^(RELIGION_CONFLICT|RELIGIOUS_CONFLICT)': '20000688',
    r'^(RELATION_RELIGION_GOVERNMENT|RELIGION_GOVERNMENT)': '20000702',
    r'^(RELIGIOUS_LEADER|IMAM|PRIEST|RABBI|PASTOR)': '20000703',

    # Politics subtopics (map to IPTC medtop qcodes)
    r'^(ELECTION|VOTE|BALLOT)': '20000574',
    r'^(GOVERNMENT|STATE|ADMINISTRATION)': '20000593',
    r'^(POLICY|GOVERNMENT_POLICY|POLICY_MAKING)': '20000621',
    r'^(DIPLOMACY|TREATY|FOREIGN_POLICY)': '20000638',
    r'^(NGO|NON_GOVERNMENT|NON-GOVERNMENTAL)': '20000646',
    r'^(POLITICAL_PRISONER|DISSENTER|POLITICAL_DISSENT)': '20000648',

    # Science & Technology subtopic mappings
    r'^(BIOMED|BIOMEDICAL|BIOMEDICAL_SCIENCE)': '20000710',
    r'^(MATH|MATHEMATICS|ALGEBRA|GEOMETRY|CALCULUS)': '20000715',
    r'^(NATURAL_SCIENCE|PHYSICS|CHEMISTRY|BIOLOGY|ASTRONOMY)': '20000717',
    r'^(SCI_INSTITUTION|RESEARCH_INSTITUTE|ACADEMY)': '20000741',
    r'^(SCIENTIFIC_RESEARCH|RESEARCH|LAB)': '20000735',
    r'^(SCIENTIFIC_STANDARD|STANDARD|STANDARDS)': '20000755',
    r'^(SOCIAL_SCIENCE|SOCIAL_SCIENCES|ECONOMICS|ANTHROPOLOGY|SOCIOLOGY)': '20000742',
    r'^(TECHNOLOGY|ENGINEERING|TECH|IT|COMPUTING)': '20000756',

    # Conflict subtopics (map to IPTC medtop qcodes)
    r'^(ACT_OF_TERROR|TERROR|TERRORISM)': '20000053',
    r'^(WAR|MILITARY|BATTLE|ARMEDCONFLICT)': '20000056',
    r'^(RIOT|DEMONSTRATION|CIVIL_UNREST|PROTEST|STRIKE)': '20000065',
    r'^(COUP|COUP_D_ETAT|COUPDETAT)': '20000070',
    r'^(CYBER|CYBER_ATTACK|CYBER_WARFARE|CYBERWAR)': '20001361',
    r'^(MASSACRE|MASS_KILLING)': '20000071',
    r'^(PEACE_PROCESS|PEACE|NEGOTIATION)': '20000073',
    r'^(POST_WAR|POST-WAR|RECONSTRUCTION|POSTWAR_RECONSTRUCTION)': '20000077',
    r'^(WAR_VICTIM|WAR_VICTIMS|VICTIM|VICTIMS)': '20001377',

    # Sport subtopics (map to IPTC medtop qcodes)
    r'^(PROFESSIONAL|ATHLETE|PLAYER)': '20001301',            # sport achievement
    r'^(OLYMPIC|AMATEUR|TOURNAMENT)': '20001108',             # sport event
    r'^(FEDERATION|ORGANIZATION|ASSOCIATION)': '20001125',    # sport organisation
    r'^(COACH|COACHING|MANAGER|COACHES)': '20001323',         # sports coaching
    r'^(TRANSFER|TRANSACTION|TRADE|DRAFT)': '20001148',       # sports transaction
    r'^(VENUE|STADIUM|ARENA|GYM)': '20001126',               # sport venue
    r'^(INDUSTRY|COMMERCIAL|COMMERCIALISATION)': '20001124',  # sport industry
    r'^(RECORD|ACHIEVEMENT|MEDAL|TITLE)': '20001301',        # sport achievement
    r'^(DOPING|DRUG|DRUG_USE|PERFORMANCE_ENHANCEMENT)': '20001104',  # drug use in sport
    r'^(DISCIPLINARY|SUSPENSION|FINE|DISCIPLINARY_ACTION)': '20001103',

    # Fallback to official top-level IDs
    # Society subtopic patterns
    r'^(COMMUNIT|COMMUNITY|COMMUNITIES)': '20000768',
    r'^(DEMOGRAPHIC|DEMOGRAPHICS|DEMOGRAPHIC_GROUP)': '20000788',
    r'^(DISCRIMINATION|DISCRIMINATE|DISCRIMINATORY)': '20000775',
    r'^(DIVERSITY|EQUITY|INCLUSION|DEI)': '20001373',
    r'^(EMIGRATION|EMIGRANT|EMIGRANTS)': '20000772',
    r'^(IMMIGRATION|IMMIGRANT|IMMIGRANTS)': '20000771',
    r'^(FAMILY|FAMILIES)': '20000780',
    r'^(RIGHTS|FUNDAMENTAL_RIGHTS)': '20000587',
    r'^(SOCIAL_CONDITION|SOCIAL_PROBLEM|WELFARE)': '20000802',
    r'^(ECON|TAX|WB|EPU|FISCAL)': '04000000',
    r'^(POLITICAL|POL|DEMOCRACY)': '11000000',
    r'^(ARMEDCONFLICT|TERROR)': '16000000',
    r'^(CORRUPTION|FRAUD)': '02000000',
    r'^(NATURAL|FAMINE)': '03000000',
    r'^(MEDICAL|DISEASE)': '07000000',
    r'^(ENV|CLIMATE)': '06000000',
    r'^(SOC|POVERTY)': '14000000',
    r'^(RELIGION|CHRISTIAN)': '12000000',
    r'^(SCIENCE|TECH)': '13000000',
    r'^(SPORT|FOOTBALL)': '15000000',
    r'^(TOURISM|TRAVEL)': '10000000',
    r'^(LABOR|EMPLOYMENT)': '09000000',
    # Weather subtopic mappings (use IPTC weather subtopics)
    r'^(METEO|METEOROLOGY|METEOROLOGICAL|METEOROLOGIST|FORECAST|WEATHER_FORECAST)': '17010000',
    r'^(STORM|HEATWAVE|TORNADO|HURRICANE|FLOOD|EXTREME_WEATHER|HEAT_WAVE|COLD_SNAP)': '17020000',
    r'^(CLIMATE_PATTERN|SEASONAL|CLIMATE|CLIMATIC|CLIMATE_TREND)': '17030000',
    r'^(WEATHER|FORECAST)': '17000000',
    r'^(HUMAN_INTEREST)': '08000000',
}

def apply_rule_based_mapping_v3(theme_code: str) -> tuple:
    """
    Apply rule-based mapping to a theme code.
    Returns (iptc_id, confidence) or (None, 0)
    """
    if not theme_code:
        return None, 0

    theme_upper = theme_code.upper()

    for pattern, iptc_id in RULE_PATTERNS_V3.items():
        if re.match(pattern, theme_upper):
            confidence = 0.9 if len(pattern) > 20 else 0.8
            return iptc_id, confidence

    return None, 0

# ============================================================================
# IPTC HIERARCHY FUNCTIONS (New)
# ============================================================================

def normalize_qcode(value: str) -> str:
    """Normalize IPTC qcode/URI to bare numeric code."""
    if not value:
        return ''
    return (value
            .replace('medtop:', '')
            .replace('http://cv.iptc.org/newscodes/mediatopic/', '')
            .replace('https://cv.iptc.org/newscodes/mediatopic/', '')
            )


def load_iptc_hierarchy(json_path: str) -> dict:
    """
    Load IPTC Media Topic hierarchy from JSON file.
    Returns dict mapping qcode to concept info including broader relationships.
    """
    import json

    hierarchy = {}
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        for concept in data.get('conceptSet', []):
            qcode = normalize_qcode(concept.get('qcode', ''))
            if qcode:
                hierarchy[qcode] = {
                    'label': concept.get('prefLabel', {}).get('en-GB', ''),
                    'definition': concept.get('definition', {}).get('en-GB', ''),
                    'broader': [normalize_qcode(b) for b in concept.get('broader', [])],
                    'narrower': [normalize_qcode(n) for n in concept.get('narrower', [])]
                }
    except Exception as e:
        print(f"[!] Could not load IPTC hierarchy: {e}")

    return hierarchy


def build_iptc_items_from_hierarchy(hierarchy: dict) -> list:
    """Flatten IPTC hierarchy into a list of items with labels/definitions for embedding."""
    items = []
    for qcode, concept in hierarchy.items():
        items.append({
            "id": qcode,
            "label": concept.get('label', ''),
            "definition": concept.get('definition', ''),
            "broader": concept.get('broader', []),
        })
    return items

def get_top_level_parent(iptc_code: str, hierarchy: dict) -> str:
    """
    Traverse up the hierarchy to find the top-level parent (one of the 17 main categories).
    Returns the top-level IPTC code.
    """
    if not iptc_code or iptc_code not in hierarchy:
        return iptc_code

    current = iptc_code
    visited = set()

    while current and current not in visited:
        visited.add(current)

        # Check if this is a top-level category (no broader or broader is empty)
        concept = hierarchy.get(current, {})
        broader = concept.get('broader', [])

        if not broader:
            # This is a top-level category
            return current

        # Move up to parent
        current = broader[0]  # Take first broader (should be only one)

    # If we can't find a top-level, return original
    return iptc_code

# Global hierarchy cache
IPTC_HIERARCHY = None

def initialize_iptc_hierarchy(data_dir: Path):
    """Initialize IPTC hierarchy from JSON file"""
    global IPTC_HIERARCHY
    if IPTC_HIERARCHY is None:
        json_path = data_dir / "cptall-en-GB.json"
        IPTC_HIERARCHY = load_iptc_hierarchy(str(json_path))
        print(f"  [OK] Loaded IPTC hierarchy with {len(IPTC_HIERARCHY)} concepts")

# ============================================================================
# LAYER 3: SUBTOPIC SIMILARITY SCORING (Modified for hierarchy)
# ============================================================================

def compute_enhanced_similarity(theme_emb, iptc_embs, iptc_items, threshold=0.6):
    """Similarity scoring against all IPTC concepts; normalize to top-level later."""
    sim_matrix = cosine_similarity(theme_emb, iptc_embs)

    results = []
    for i in range(len(sim_matrix)):
        scores = sim_matrix[i]

        best_idx = np.argmax(scores)
        best_score = scores[best_idx]
        best_item = iptc_items[best_idx]

        top_level_id = get_top_level_parent(best_item["id"], IPTC_HIERARCHY)

        results.append({
            "final_id": top_level_id,
            "final_label": get_iptc_label(top_level_id),
            "confidence": float(best_score),
            "level": "nn-best",
            "top_id": top_level_id,
            "top_label": get_iptc_label(top_level_id),
            "sub_id": best_item["id"],
            "sub_label": best_item.get("label", ""),
            "sub_definition": best_item.get("definition", ""),
        })

    return results

# ============================================================================
# FUSION: HIERARCHICAL DECISION TREE
# ============================================================================

THR_STRONG_V3 = 0.45  # NN acceptance threshold (stricter to avoid weak matches)
THR_WEAK_V3 = 0.15    # Legacy weak threshold (unused in new fusion)

def fusion_decision_v3(rule_id: str, rule_conf: float, nn_result: dict, iptc_items: list) -> tuple:
    """Fusion where NN has priority; rules are last-resort when NN is below threshold."""
    nn_id = nn_result.get("final_id")
    nn_conf = nn_result.get("confidence", 0)

    # First: accept NN if strong enough
    if nn_id and nn_conf >= THR_STRONG_V3:
        top_level_nn_id = get_top_level_parent(nn_id, IPTC_HIERARCHY)
        return top_level_nn_id, 'nn', nn_conf, {
            "top_id": top_level_nn_id,
            "top_label": get_iptc_label(top_level_nn_id),
            "sub_id": nn_result.get('sub_id'),
            "sub_label": nn_result.get('sub_label', ''),
            "sub_definition": nn_result.get('sub_definition', '')
        }

    # Second: use rules only as a last resort when NN is weak
    if rule_id and nn_conf < THR_STRONG_V3:
        top_level_rule_id = get_top_level_parent(rule_id, IPTC_HIERARCHY)
        return top_level_rule_id, 'rule_fallback', rule_conf, {
            "top_id": top_level_rule_id,
            "top_label": get_iptc_label(top_level_rule_id),
            "sub_id": None,
            "sub_label": "",
            "sub_definition": ""
        }

    # Otherwise unclassified
    return None, 'unclassified', nn_conf, {
        "top_id": None,
        "top_label": "",
        "sub_id": None,
        "sub_label": "",
        "sub_definition": ""
    }

# ============================================================================
# MAIN PIPELINE (Modified for v3)
# ============================================================================

def run_full_mapping_v3(data_dir: str = ".", output_dir: str = ".", mapping_source: str = 'vargo') -> dict:
    """
    Run the complete three-layer GDELT -> IPTC mapping pipeline with top-level categories only.

    Returns:
        Dictionary with mapping results and statistics
    """
    print("=" * 60)
    print("GDELT Theme -> IPTC Media Topics Mapper (v3)")
    print("Full Hierarchy Embedding -> Top-level Normalization")
    print("=" * 60)

    output_path = Path(output_dir)
    data_path = Path(data_dir)
    results_path = Path(output_dir)  # Results are saved to output_dir

    # Ensure output directory exists (use relative paths when possible)
    try:
        output_path.mkdir(parents=True, exist_ok=True)
    except Exception:
        # Fallback: create relative 'results' directory next to script
        output_path = Path(__file__).parent / "results"
        output_path.mkdir(parents=True, exist_ok=True)

    # Initialize IPTC hierarchy
    print("\n[*] Loading IPTC Media Topic hierarchy...")
    initialize_iptc_hierarchy(data_path)

    # Load theme codes from all CSV files
    print("\n[*] Loading theme codes from CSV files...")

    theme_codes = set()
    csv_files = [
        'gdelt_top15_themes_by_country_2022_2024.csv',
        'gdelt_monthly_quality_metrics.csv',
        'gdelt_monthly_docs_per_theme_country_2022_2024.csv'
    ]

    # Input CSVs should be read from the project's `data/` directory (relative path)
    for csv_file in csv_files:
        try:
            df = pd.read_csv(data_path / csv_file)
            if 'theme_code' in df.columns:
                theme_codes.update(df['theme_code'].dropna().unique())
        except Exception as e:
            print(f"  [!] Could not read {csv_file}: {e}")

    if not theme_codes:
        print("[X] No theme codes found in CSV files!")
        return None

    print(f"  [OK] Found {len(theme_codes)} unique theme codes")

    # Load mapping source
    print(f"\n[*] Loading {mapping_source.upper()} mapping...")
    mapping_data = load_mapping_source(data_path, mapping_source)
    print(f"  [OK] Loaded {len(mapping_data)} {mapping_source.upper()} mappings")

    # LAYER 1: Rule-based mapping (extended for subtopics)
    print("\n[*] Layer 1: Applying rule-based mapping...")
    rule_results = {}
    rule_matched = 0

    for theme in theme_codes:
        iptc_id, confidence = apply_rule_based_mapping_v3(theme)
        rule_results[theme] = {'rule_id': iptc_id, 'rule_conf': confidence}
        if iptc_id:
            rule_matched += 1

    print(f"  [OK] Rule-based matched: {rule_matched}/{len(theme_codes)} ({100*rule_matched/len(theme_codes):.1f}%)")

    # LAYER 2 & 3: Embedding-based mapping with top-level categories only
    if not TRANSFORMERS_AVAILABLE:
        print("\n[!] Sentence-transformers not available. Using rule-based only.")
        nn_results = {theme: {
            'nn_id': None, 'nn_label': '', 'nn_score': 0,
            'level': 'unclassified', 'top_id': None, 'sub_id': None
        } for theme in theme_codes}
    else:
        print("\n[*] Layer 2-3: Computing semantic embeddings with full IPTC hierarchy...")

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"  Device: {device}")

        print("  Loading sentence-transformer model...")
        model = SentenceTransformer('all-MiniLM-L6-v2', device=device)  # Back to proven model

        # Build text representations - only top-level categories for better matching
        theme_list = sorted(theme_codes)
        theme_texts = [build_theme_text(t, mapping_data, mapping_source) for t in theme_list]
        
        # Encode all IPTC concepts from hierarchy (prefLabel + definition)
        iptc_items = build_iptc_items_from_hierarchy(IPTC_HIERARCHY)
        iptc_texts = [(item.get('label', '') + " " + item.get('definition', '')).strip() or item.get('id', '') for item in iptc_items]

        print(f"  Encoding {len(theme_texts)} themes...")
        theme_embeddings = compute_embeddings(theme_texts, model)

        print(f"  Encoding {len(iptc_texts)} IPTC concepts (full hierarchy)...")
        iptc_embeddings = compute_embeddings(iptc_texts, model)

        print("  Computing similarities...")
        nn_list = compute_enhanced_similarity(theme_embeddings, iptc_embeddings, iptc_items)

        # Debug: Print similarity statistics
        confidences = [r['confidence'] for r in nn_list]
        print(f"  Similarity stats: min={min(confidences):.3f}, max={max(confidences):.3f}, mean={np.mean(confidences):.3f}, median={np.median(confidences):.3f}")
        print(f"  Thresholds: strong={THR_STRONG_V3}, weak={THR_WEAK_V3}")

        nn_results = {theme: nn_list[i] for i, theme in enumerate(theme_list)}

        # Compute 2D projections for visualization
        print("\n[*] Computing 2D projections for visualization...")
        coords_2d = compute_2d_projection(theme_embeddings, method='tsne')

        # Store coordinates in nn_results
        for i, theme in enumerate(theme_list):
            nn_results[theme]['tsne_x'] = float(coords_2d[i, 0])
            nn_results[theme]['tsne_y'] = float(coords_2d[i, 1])

    # FUSION
    print("\n[*] Fusion: Combining rule-based and hierarchical embedding results...")

    final_results = []
    decision_counts = {}

    for theme in sorted(theme_codes):
        rule_id = rule_results[theme]['rule_id']
        rule_conf = rule_results[theme]['rule_conf']

        nn_data = nn_results.get(theme, {
            'final_id': None, 'final_label': '', 'confidence': 0,
            'level': 'unclassified', 'top_id': None, 'sub_id': None
        })

        final_id, decision, confidence, subtopic_info = fusion_decision_v3(
            rule_id, rule_conf, nn_data, IPTC_ALL_ITEMS
        )

        decision_counts[decision] = decision_counts.get(decision, 0) + 1

        result = {
            'theme_code': theme,
            'text_repr': build_theme_text(theme, mapping_data, mapping_source),
            'iptc_rule_id': rule_id or '',
            'iptc_rule_label': get_iptc_label(rule_id) if rule_id else '',
            'rule_confidence': rule_conf,
            'iptc_nn_id': nn_data.get('final_id') or '',
            'iptc_nn_label': nn_data.get('final_label', ''),
            'nn_score': nn_data.get('confidence', 0),
            'iptc_final_id': final_id or '',
            'iptc_final_label': get_iptc_label(final_id) if final_id else '',
            'decision_source': decision,
            'final_confidence': confidence,
            # Subtopic information
            'subtopic_info': subtopic_info,
            # 2D coordinates for visualization
            'tsne_x': nn_data.get('tsne_x', 50 + np.random.randn() * 10),
            'tsne_y': nn_data.get('tsne_y', 50 + np.random.randn() * 10)
        }
        final_results.append(result)

    # Statistics
    print("\n[*] Fusion Statistics:")
    for decision, count in sorted(decision_counts.items(), key=lambda x: -x[1]):
        pct = 100 * count / len(theme_codes)
        print(f"  {decision}: {count} ({pct:.1f}%)")

    # Category distribution (top-level only for compatibility)
    category_dist = {}
    for r in final_results:
        cat = r['iptc_final_label'] or 'unclassified'
        category_dist[cat] = category_dist.get(cat, 0) + 1

    print("\n[*] IPTC Category Distribution (Top-level):")
    for cat, count in sorted(category_dist.items(), key=lambda x: -x[1]):
        print(f"  {cat}: {count}")

    # Save results
    print("\n[*] Saving results...")

    # Build output structure (maintain v2 compatibility)
    output_data = {
        "metadata": {
            "generated_at": datetime.now().isoformat(),
            "total_themes": len(theme_codes),
            "rule_matched": rule_matched,
            "iptc_categories": len(set(r['iptc_final_label'] for r in final_results if r['iptc_final_label'])),
            "decision_stats": decision_counts,
            "mapping_source": mapping_source,
            "version": "v3",
            "subtopics_enabled": False
        },
        "themes": final_results,
        "iptc_categories": {}
    }

    # Group by IPTC category (top-level for compatibility)
    for r in final_results:
        cat_label = r['iptc_final_label'] or 'unclassified'
        if cat_label not in output_data["iptc_categories"]:
            output_data["iptc_categories"][cat_label] = {
                "iptc_id": r['iptc_final_id'] or '',
                "theme_count": 0,
                "themes": []
            }
        output_data["iptc_categories"][cat_label]["theme_count"] += 1
        output_data["iptc_categories"][cat_label]["themes"].append(r['theme_code'])

    # Clean data for JSON serialization
    def clean_for_json(obj):
        if isinstance(obj, dict):
            return {k: clean_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [clean_for_json(v) for v in obj]
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj

    output_data = clean_for_json(output_data)

    # Save JSON
    json_filename = f"gdelt_iptc_mapping_v3_{mapping_source}.json"
    json_path = output_path / json_filename
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    print(f"  [OK] JSON: {json_path.name}")

    # Save CSV
    csv_filename = f"gdelt_themes_iptc_v3_{mapping_source}.csv"
    csv_path = output_path / csv_filename
    # Ensure IPTC id columns are strings to preserve leading zeros in CSV exports
    df_out = pd.DataFrame(final_results)
    for col in ['iptc_rule_id', 'iptc_nn_id', 'iptc_final_id']:
        if col in df_out.columns:
            df_out[col] = (
                df_out[col]
                .fillna('')
                .astype(str)
                .str.replace(r'\.0+$', '', regex=True)  # strip float artefacts
                .str.split('.').str[0]                   # guard against scientific notation
            )

    df_out.to_csv(csv_path, index=False)
    print(f"  [OK] CSV: {csv_path.name}")

    print("\n[OK] V3 Mapping complete!")
    print(f"\n[*] Summary:")
    print(f"   Total themes: {len(theme_codes)}")
    top_matches = decision_counts.get('nn', 0) + decision_counts.get('rule_fallback', 0)
    print(f"   Top-level matches: {top_matches}")
    print(f"   Unclassified: {decision_counts.get('unclassified', 0)}")

    return output_data

# ============================================================================
# UTILITY FUNCTIONS (from v2)
# ============================================================================

def load_vargo_mapping(filepath: str = "vargo_gdelt_themes_issues.csv") -> dict:
    """Load Vargo's GDELT theme to issue mapping"""
    vargo_map = {}
    try:
        df = pd.read_csv(filepath)
        for _, row in df.iterrows():
            vargo_map[row['theme_code']] = {
                'issue': row['issue_category'],
                'description': row.get('description', '')
            }
    except Exception as e:
        print(f"[!] Could not load Vargo mapping: {e}")
    return vargo_map


def load_gkg_mapping(filepath: str = "gdelt_gkg_categorylist.csv") -> dict:
    """Load GKG Category List mapping - filtered to Theme entries only"""
    gkg_map = {}
    try:
        df = pd.read_csv(filepath)
        # Filter to only Theme entries
        theme_df = df[df['Type'] == 'Theme']
        for _, row in theme_df.iterrows():
            gkg_map[row['Name']] = {
                'description': row.get('Description', '')
            }
    except Exception as e:
        print(f"[!] Could not load GKG mapping: {e}")
    return gkg_map


def load_combined_mapping(data_dir: Path) -> dict:
    """Load combined Vargo + GKG mapping for richer descriptions"""
    combined_map = {}

    # Load Vargo mapping
    vargo_map = load_vargo_mapping(data_dir / "vargo_gdelt_themes_issues.csv")

    # Load GKG mapping
    gkg_map = load_gkg_mapping(data_dir / "gdelt_gkg_categorylist.csv")

    # Combine mappings
    all_theme_codes = set(vargo_map.keys()) | set(gkg_map.keys())

    for theme_code in all_theme_codes:
        descriptions = []

        # Add GKG description if available
        if theme_code in gkg_map and gkg_map[theme_code].get('description'):
            descriptions.append(gkg_map[theme_code]['description'])

        # Add Vargo description if available
        if theme_code in vargo_map:
            vargo_info = vargo_map[theme_code]
            if vargo_info.get('description'):
                descriptions.append(vargo_info['description'])
            if vargo_info.get('issue'):
                descriptions.append(f"Issue: {vargo_info['issue']}")

        # Combine all descriptions
        combined_description = ' | '.join(descriptions) if descriptions else ''

        combined_map[theme_code] = {
            'description': combined_description,
            'sources': []
        }

        if theme_code in gkg_map:
            combined_map[theme_code]['sources'].append('gkg')
        if theme_code in vargo_map:
            combined_map[theme_code]['sources'].append('vargo')

    print(f"  [OK] Combined {len(combined_map)} mappings from Vargo ({len(vargo_map)}) + GKG ({len(gkg_map)})")

    return combined_map


def load_mapping_source(data_dir: Path, mapping_source: str) -> dict:
    """Load mapping source based on selection."""
    if mapping_source == 'vargo':
        return load_vargo_mapping(data_dir / "vargo_gdelt_themes_issues.csv")
    elif mapping_source == 'gkg':
        return load_gkg_mapping(data_dir / "gdelt_gkg_categorylist.csv")
    elif mapping_source == 'combined':
        return load_combined_mapping(data_dir)
    else:
        print(f"[!] Unknown mapping source: {mapping_source}, using combined as default")
        return load_combined_mapping(data_dir)


def build_theme_text(theme_code: str, mapping_data: dict, mapping_source: str) -> str:
    """Build text representation using only descriptions for semantic comparison"""
    if theme_code in mapping_data:
        info = mapping_data[theme_code]
        description = info.get('description', '')
        if description:
            return f"{theme_code} - {description}"
    # Fallback if no description
    return theme_code.replace('_', ' ').replace('-', ' ').lower()


def build_iptc_texts(categories: list) -> list:
    """Build text representations for IPTC categories"""
    return [f"{cat['label']} - {cat['definition']}" for cat in categories]


def compute_embeddings(texts: list, model, batch_size: int = 64) -> np.ndarray:
    """Compute sentence embeddings"""
    return model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True
    )


def find_nearest_iptc(theme_emb: np.ndarray, iptc_emb: np.ndarray, categories: list) -> list:
    """Find nearest IPTC category for each theme embedding"""
    sim = cosine_similarity(theme_emb, iptc_emb)

    results = []
    for i in range(len(sim)):
        scores = sim[i]
        best_idx = np.argmax(scores)
        second_best_idx = np.argsort(scores)[-2]

        results.append({
            'nn_id': categories[best_idx]['id'],
            'nn_label': categories[best_idx]['label'],
            'nn_score': float(scores[best_idx]),
            'second_id': categories[second_best_idx]['id'],
            'second_label': categories[second_best_idx]['label'],
            'second_score': float(scores[second_best_idx])
        })

    return results


def compute_2d_projection(embeddings: np.ndarray, method: str = 'tsne') -> np.ndarray:
    """
    Compute 2D projection of embeddings using t-SNE or PCA.

    Args:
        embeddings: High-dimensional embeddings (N x D)
        method: 'tsne' or 'pca'

    Returns:
        2D coordinates (N x 2)
    """
    n_samples = len(embeddings)

    if n_samples < 5:
        print(f"  [!] Too few samples ({n_samples}) for dimensionality reduction")
        return np.random.randn(n_samples, 2) * 10

    if method == 'tsne':
        # t-SNE parameters optimized for theme clustering visualization
        perplexity = min(30, n_samples - 1)  # perplexity must be < n_samples
        print(f"  Running t-SNE (perplexity={perplexity})...")

        # First reduce with PCA if high dimensional (faster t-SNE)
        if embeddings.shape[1] > 50:
            n_pca = min(50, n_samples - 1)
            pca = PCA(n_components=n_pca, random_state=42)
            embeddings_reduced = pca.fit_transform(embeddings)
            print(f"  PCA pre-reduction: {embeddings.shape[1]} -> {n_pca} dims")
        else:
            embeddings_reduced = embeddings

        tsne = TSNE(
            n_components=2,
            perplexity=perplexity,
            learning_rate='auto',
            init='pca' if n_samples > 50 else 'random',
            random_state=42,
            max_iter=1000
        )
        coords_2d = tsne.fit_transform(embeddings_reduced)

    else:  # PCA
        print("  Running PCA...")
        pca = PCA(n_components=2, random_state=42)
        coords_2d = pca.fit_transform(embeddings)

    # Normalize to 0-100 range for visualization
    coords_min = coords_2d.min(axis=0)
    coords_max = coords_2d.max(axis=0)
    coords_range = coords_max - coords_min
    coords_range[coords_range == 0] = 1  # Avoid division by zero

    coords_normalized = 5 + 90 * (coords_2d - coords_min) / coords_range  # 5-95 range

    return coords_normalized


def get_iptc_label(iptc_id: str) -> str:
    """Get IPTC category label from ID"""
    for cat in IPTC_CATEGORIES_V3:
        if cat['id'] == iptc_id:
            return cat['label']
        for sub in cat.get('subtopics', []):
            if sub['id'] == iptc_id:
                return f"{cat['label']} - {sub['label']}"
    if IPTC_HIERARCHY and iptc_id in IPTC_HIERARCHY:
        return IPTC_HIERARCHY[iptc_id].get('label', '')
    return ''

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Map GDELT themes to IPTC Media Topics (V3 - Top-level Categories Only)"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Directory containing data files"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save output files"
    )
    parser.add_argument(
        "--mapping-source",
        type=str,
        choices=['vargo', 'gkg', 'combined'],
        default='combined',
        help="Mapping source to use: 'vargo' (Vargo issue taxonomy), 'gkg' (GKG Category List), or 'combined' (both)"
    )

    args = parser.parse_args()

    data_dir = Path(args.data_dir) if args.data_dir else Path(__file__).parent / "data"
    output_dir = Path(args.output_dir) if args.output_dir else Path(__file__).parent / "results"

    results = run_full_mapping_v3(data_dir=data_dir, output_dir=output_dir, mapping_source=args.mapping_source)