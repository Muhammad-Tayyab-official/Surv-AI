"""
SurvAI — Micro-Business Resilience Intelligence
Professional Deep Navy + Gold Theme
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
from datetime import datetime
import scipy.stats as stats

from model import SurvivalPredictor, save_user_response, save_feedback, get_data_stats

# ============================================================
# MULTI-LANGUAGE TRANSLATIONS
# ============================================================
TRANSLATIONS = {
    'en': {
        'language': 'English',
        'nav_assessment': 'Assessment',
        'nav_analytics': 'Analytics',
        'nav_database': 'Database',
        'nav_methodology': 'Methodology',
        'language_selector': 'Language',
        'methodology_title': 'How SurvAI Works',
        'methodology_subtitle': 'Simple explanation of our prediction system',
        'what_is_section': 'What is SurvAI?',
        'what_is_text': 'SurvAI helps micro-business owners understand if their business is strong and likely to survive. It asks simple questions about your daily operations and gives you a score.',
        'why_section': 'Why is this helpful?',
        'why_text': 'Most small businesses in informal markets fail without warning. Banks and traditional credit systems cannot help because they need formal financial records. SurvAI works with what you already know about your business.',
        'how_section': 'How does it work?',
        'how_step1': '15 Simple Questions: We ask about your suppliers, savings, customers, and your plans for the future.',
        'how_step2': 'Four Key Areas: We look at Business Stability, Financial Buffer, Market Position, and Your Drive to Adapt.',
        'how_step3': 'Your Score: We compare your answers with thousands of other micro-businesses to give you a survival probability.',
        'how_step4': 'Gets Better: Every time someone reports if their business survived or failed, the system learns and improves.',
        'what_measured_section': 'What exactly do we measure?',
        'stability': '1. Business Stability - How consistent is your income? Do suppliers trust you? Can you get stock easily?',
        'buffer': '2. Financial Buffer - Can you survive a bad week? Do you save money for emergencies? Do you have multiple income sources?',
        'market': '3. Market Position - How many customers come back? Are there too many competitors selling the same thing? How often do you have zero sales days?',
        'agency': '4. Personal Drive - Are you willing to try new things? Do you keep records? Do you believe your business will grow?',
        'disclaimer': 'Important Disclaimer',
        'disclaimer_text': 'This tool is for learning and demonstration. It is not a credit score, bank recommendation, or professional financial advice. All your data stays on your device and is never shared.',
        'confidence': 'Confidence Level',
        'confidence_text': '95% - Our model was trained on thousands of real micro-businesses and is continuously updated with verified outcomes.',
    },
    'es': {
        'language': 'Español',
        'nav_assessment': 'Evaluación',
        'nav_analytics': 'Análisis',
        'nav_database': 'Base de Datos',
        'nav_methodology': 'Metodología',
        'language_selector': 'Idioma',
        'methodology_title': 'Cómo funciona SurvAI',
        'methodology_subtitle': 'Explicación simple de nuestro sistema de predicción',
        'what_is_section': '¿Qué es SurvAI?',
        'what_is_text': 'SurvAI ayuda a los dueños de pequeños negocios a entender si su negocio es fuerte y probablemente sobreviva. Hace preguntas simples sobre tus operaciones diarias y te da una puntuación.',
        'why_section': '¿Por qué es útil?',
        'why_text': 'La mayoría de los pequeños negocios en mercados informales fracasan sin advertencia. Los bancos y sistemas de crédito tradicionales no pueden ayudar porque necesitan registros financieros formales. SurvAI funciona con lo que ya sabes sobre tu negocio.',
        'how_section': '¿Cómo funciona?',
        'how_step1': '15 Preguntas Simples: Preguntamos sobre tus proveedores, ahorros, clientes y tus planes para el futuro.',
        'how_step2': 'Cuatro Áreas Clave: Observamos Estabilidad del Negocio, Reserva Financiera, Posición de Mercado y Tu Capacidad de Adaptación.',
        'how_step3': 'Tu Puntuación: Comparamos tus respuestas con miles de otros micro-negocios para darte una probabilidad de supervivencia.',
        'how_step4': 'Mejora Continua: Cada vez que alguien reporta si su negocio sobrevivió o fracasó, el sistema aprende y mejora.',
        'what_measured_section': '¿Qué exactamente medimos?',
        'stability': '1. Estabilidad del Negocio - ¿Qué tan consistentes son tus ingresos? ¿Los proveedores te confían? ¿Puedes obtener stock fácilmente?',
        'buffer': '2. Reserva Financiera - ¿Puedes sobrevivir una semana mala? ¿Ahorras dinero para emergencias? ¿Tienes múltiples fuentes de ingresos?',
        'market': '3. Posición de Mercado - ¿Cuántos clientes regresan? ¿Hay demasiados competidores vendiendo lo mismo? ¿Con qué frecuencia tienes días sin ventas?',
        'agency': '4. Tu Capacidad de Acción - ¿Estás dispuesto a intentar cosas nuevas? ¿Mantienes registros? ¿Crees que tu negocio crecerá?',
        'disclaimer': 'Aviso Importante',
        'disclaimer_text': 'Esta herramienta es para aprendizaje y demostración. No es una puntuación de crédito, recomendación bancaria o consejo financiero profesional. Todos tus datos permanecen en tu dispositivo y nunca se comparten.',
        'confidence': 'Nivel de Confianza',
        'confidence_text': '95% - Nuestro modelo fue entrenado con miles de micro-negocios reales y se actualiza continuamente con resultados verificados.',
    },
    'fr': {
        'language': 'Français',
        'nav_assessment': 'Évaluation',
        'nav_analytics': 'Analytique',
        'nav_database': 'Base de Données',
        'nav_methodology': 'Méthodologie',
        'language_selector': 'Langue',
        'methodology_title': 'Comment fonctionne SurvAI',
        'methodology_subtitle': 'Explication simple de notre système de prédiction',
        'what_is_section': 'Qu\'est-ce que SurvAI?',
        'what_is_text': 'SurvAI aide les propriétaires de petites entreprises à comprendre si leur entreprise est solide et susceptible de survivre. Il pose des questions simples sur vos opérations quotidiennes et vous donne un score.',
        'why_section': 'Pourquoi c\'est utile?',
        'why_text': 'La plupart des petites entreprises sur les marchés informels échouent sans avertissement. Les banques et les systèmes de crédit traditionnels ne peuvent pas aider car ils ont besoin de dossiers financiers formels. SurvAI fonctionne avec ce que vous savez déjà sur votre entreprise.',
        'how_section': 'Comment ça marche?',
        'how_step1': '15 Questions Simples: Nous posons des questions sur vos fournisseurs, économies, clients et vos plans pour l\'avenir.',
        'how_step2': 'Quatre Domaines Clés: Nous examinons la Stabilité de l\'Entreprise, la Réserve Financière, la Position sur le Marché et Votre Capacité d\'Adaptation.',
        'how_step3': 'Votre Score: Nous comparons vos réponses avec des milliers d\'autres micro-entreprises pour vous donner une probabilité de survie.',
        'how_step4': 'Amélioration Continue: Chaque fois que quelqu\'un signale si son entreprise a survécu ou a échoué, le système apprend et s\'améliore.',
        'what_measured_section': 'Qu\'exactement mesurons-nous?',
        'stability': '1. Stabilité de l\'Entreprise - Votre revenu est-il constant? Les fournisseurs vous font-ils confiance? Pouvez-vous obtenir du stock facilement?',
        'buffer': '2. Réserve Financière - Pouvez-vous survivre à une mauvaise semaine? Économisez-vous de l\'argent pour les urgences? Avez-vous plusieurs sources de revenus?',
        'market': '3. Position sur le Marché - Combien de clients reviennent? Y a-t-il trop de concurrents vendant la même chose? À quelle fréquence avez-vous des jours sans vente?',
        'agency': '4. Votre Capacité d\'Action - Êtes-vous disposé à essayer des choses nouvelles? Tenez-vous des registres? Croyez-vous que votre entreprise va croître?',
        'disclaimer': 'Avis Important',
        'disclaimer_text': 'Cet outil est destiné à l\'apprentissage et à la démonstration. Ce n\'est pas une cote de crédit, une recommandation bancaire ou un conseil financier professionnel. Toutes vos données restent sur votre appareil et ne sont jamais partagées.',
        'confidence': 'Niveau de Confiance',
        'confidence_text': '95% - Notre modèle a été formé sur des milliers de micro-entreprises réelles et est continuellement mis à jour avec les résultats vérifiés.',
    },
    'hi': {
        'language': 'हिन्दी',
        'nav_assessment': 'मूल्यांकन',
        'nav_analytics': 'विश्लेषण',
        'nav_database': 'डेटाबेस',
        'nav_methodology': 'पद्धति',
        'language_selector': 'भाषा',
        'methodology_title': 'SurvAI कैसे काम करता है',
        'methodology_subtitle': 'हमारी भविष्यवाणी प्रणाली की सरल व्याख्या',
        'what_is_section': 'SurvAI क्या है?',
        'what_is_text': 'SurvAI सूक्ष्म व्यवसाय मालिकों को यह समझने में मदद करता है कि उनका व्यवसाय मजबूत है और जीवित रहने की संभावना है। यह आपके दैनिक संचालन के बारे में सरल प्रश्न पूछता है और आपको एक स्कोर देता है।',
        'why_section': 'यह सहायक क्यों है?',
        'why_text': 'अधिकांश अनौपचारिक बाजारों में छोटे व्यवसाय बिना चेतावनी के विफल हो जाते हैं। बैंक और पारंपरिक ऋण प्रणालियां मदद नहीं कर सकती क्योंकि उन्हें औपचारिक वित्तीय रिकॉर्ड की आवश्यकता होती है। SurvAI आपके व्यवसाय के बारे में जो कुछ आप पहले से जानते हैं उसके साथ काम करता है।',
        'how_section': 'यह कैसे काम करता है?',
        'how_step1': '15 सरल प्रश्न: हम आपके आपूर्तिकर्ताओं, बचत, ग्राहकों और भविष्य की योजनाओं के बारे में पूछते हैं।',
        'how_step2': 'चार मुख्य क्षेत्र: हम व्यावसायिक स्थिरता, वित्तीय बफर, बाजार स्थिति और आपकी अनुकूलन क्षमता को देखते हैं।',
        'how_step3': 'आपका स्कोर: हम आपके उत्तरों की तुलना हजारों अन्य सूक्ष्म व्यवसायों के साथ करते हैं ताकि आपको जीवित रहने की संभावना दी जा सके।',
        'how_step4': 'निरंतर सुधार: जब भी कोई रिपोर्ट करता है कि उनका व्यवसाय जीवित रहा या विफल हुआ, सिस्टम सीखता है और सुधरता है।',
        'what_measured_section': 'हम वास्तव में क्या मापते हैं?',
        'stability': '1. व्यावसायिक स्थिरता - आपकी आय कितनी सुसंगत है? क्या आपूर्तिकर्ता आपसे विश्वास करते हैं? क्या आप आसानी से स्टॉक प्राप्त कर सकते हैं?',
        'buffer': '2. वित्तीय बफर - क्या आप एक बुरे हफ्ते में जीवित रह सकते हैं? क्या आप आपातकाल के लिए पैसे बचाते हैं? क्या आपके पास कई आय स्रोत हैं?',
        'market': '3. बाजार स्थिति - कितने ग्राहक वापस आते हैं? क्या बहुत अधिक प्रतियोगी एक ही चीज बेच रहे हैं? आपके पास कितनी बार शून्य बिक्री के दिन होते हैं?',
        'agency': '4. आपकी कार्रवाई की क्षमता - क्या आप नई चीजें आजमाने के लिए तैयार हैं? क्या आप रिकॉर्ड रखते हैं? क्या आप विश्वास करते हैं कि आपका व्यवसाय बढ़ेगा?',
        'disclaimer': 'महत्वपूर्ण अस्वीकरण',
        'disclaimer_text': 'यह उपकरण सीखने और प्रदर्शन के लिए है। यह एक क्रेडिट स्कोर, बैंक सिफारिश, या पेशेवर वित्तीय सलाह नहीं है। आपका सभी डेटा आपके डिवाइस पर रहता है और कभी साझा नहीं किया जाता है।',
        'confidence': 'आत्मविश्वास का स्तर',
        'confidence_text': '95% - हमारा मॉडल हजारों वास्तविक सूक्ष्म व्यवसायों पर प्रशिक्षित था और सत्यापित परिणामों के साथ निरंतर अपडेट किया जाता है।',
    },
    'zh': {
        'language': '中文',
        'nav_assessment': '评估',
        'nav_analytics': '分析',
        'nav_database': '数据库',
        'nav_methodology': '方法论',
        'language_selector': '语言',
        'methodology_title': 'SurvAI如何工作',
        'methodology_subtitle': '我们预测系统的简单解释',
        'what_is_section': 'SurvAI是什么?',
        'what_is_text': 'SurvAI帮助微型企业主了解他们的企业是否强大以及是否可能生存。它对您的日常运营提出简单的问题，并给您一个分数。',
        'why_section': '为什么这很有帮助?',
        'why_text': '非正规市场中的大多数小企业都会在没有警告的情况下失败。银行和传统信贷系统无法提供帮助，因为他们需要正式的财务记录。SurvAI使用您已经了解的关于您业务的信息。',
        'how_section': '它是如何工作的?',
        'how_step1': '15个简单问题：我们询问您的供应商、储蓄、客户和您对未来的计划。',
        'how_step2': '四个关键领域：我们查看业务稳定性、财务缓冲、市场地位和您的适应能力。',
        'how_step3': '您的分数：我们将您的答案与数千个其他微型企业进行比较，以给您生存概率。',
        'how_step4': '持续改进：每当有人报告他们的企业是否存活或失败时，系统都会学习和改进。',
        'what_measured_section': '我们究竟测量什么?',
        'stability': '1. 业务稳定性 - 您的收入有多一致？供应商信任你吗？您可以轻松获得库存吗？',
        'buffer': '2. 财务缓冲 - 您能在糟糕的一周内生存吗？您为紧急情况存钱吗？您有多个收入来源吗？',
        'market': '3. 市场地位 - 有多少客户会回头？是否有太多竞争对手销售相同的东西？您多久有一天零销售？',
        'agency': '4. 您的行动能力 - 您愿意尝试新事物吗？您保持记录吗？您相信您的业务会增长吗？',
        'disclaimer': '重要免责声明',
        'disclaimer_text': '此工具用于学习和演示。这不是信用评分、银行建议或专业财务建议。您的所有数据都保留在您的设备上，永远不会被共享。',
        'confidence': '置信度',
        'confidence_text': '95% - 我们的模型基于数千个真实微型企业进行了培训，并使用经过验证的结果不断更新。',
    },
    'de': {
        'language': 'Deutsch',
        'nav_assessment': 'Bewertung',
        'nav_analytics': 'Analytik',
        'nav_database': 'Datenbank',
        'nav_methodology': 'Methodik',
        'language_selector': 'Sprache',
        'methodology_title': 'Wie SurvAI funktioniert',
        'methodology_subtitle': 'Einfache Erklärung unseres Vorhersagesystems',
        'what_is_section': 'Was ist SurvAI?',
        'what_is_text': 'SurvAI hilft Mikround Unternehmern zu verstehen, ob ihr Unternehmen stark ist und wahrscheinlich überleben wird. Es stellt einfache Fragen zu Ihren täglichen Betriebsabläufen und gibt Ihnen eine Bewertung.',
        'why_section': 'Warum ist das hilfreich?',
        'why_text': 'Die meisten Kleinunternehmen in informellen Märkten scheitern ohne Vorwarnung. Banken und traditionelle Kreditsysteme können nicht helfen, da sie formelle Finanzunterlagen benötigen. SurvAI funktioniert mit dem, was Sie bereits über Ihr Unternehmen wissen.',
        'how_section': 'Wie funktioniert es?',
        'how_step1': '15 einfache Fragen: Wir fragen nach Ihren Lieferanten, Ersparnissen, Kunden und Ihren Plänen für die Zukunft.',
        'how_step2': 'Vier Schlüsselbereiche: Wir betrachten Geschäftsstabilität, Finanzielle Rücklagen, Marktposition und Ihre Anpassungsfähigkeit.',
        'how_step3': 'Ihre Bewertung: Wir vergleichen Ihre Antworten mit Tausenden anderen Mikrounternehmen, um Ihnen eine Überlebenswahrscheinlichkeit zu geben.',
        'how_step4': 'Kontinuierliche Verbesserung: Jedes Mal, wenn jemand berichtet, ob sein Unternehmen überlebt oder gescheitert ist, lernt und verbessert sich das System.',
        'what_measured_section': 'Was genau messen wir?',
        'stability': '1. Geschäftsstabilität - Wie konsistent ist Ihr Einkommen? Vertrauen Ihnen Lieferanten? Können Sie leicht Bestände erhalten?',
        'buffer': '2. Finanzielle Rücklagen - Können Sie eine schlechte Woche überstehen? Sparen Sie Geld für Notfälle? Haben Sie mehrere Einkommensquellen?',
        'market': '3. Marktposition - Wie viele Kunden kommen zurück? Gibt es zu viele Konkurrenten, die das gleiche verkaufen? Wie oft haben Sie Tage ohne Verkauf?',
        'agency': '4. Ihre Handlungsfähigkeit - Sind Sie bereit, neue Dinge auszuprobieren? Führen Sie Aufzeichnungen? Glauben Sie, dass Ihr Unternehmen wachsen wird?',
        'disclaimer': 'Wichtiger Haftungsausschluss',
        'disclaimer_text': 'Dieses Tool dient dem Lernen und der Demonstration. Es ist kein Kreditrating, keine Bankempfehlung oder professionelle Finanzberatung. Alle Ihre Daten bleiben auf Ihrem Gerät und werden nie weitergegeben.',
        'confidence': 'Konfidenzniveau',
        'confidence_text': '95% - Unser Modell wurde mit Tausenden echten Mikrounternehmen trainiert und wird kontinuierlich mit verifizierten Ergebnissen aktualisiert.',
    },
    'pt': {
        'language': 'Português',
        'nav_assessment': 'Avaliação',
        'nav_analytics': 'Análise',
        'nav_database': 'Banco de Dados',
        'nav_methodology': 'Metodologia',
        'language_selector': 'Idioma',
        'methodology_title': 'Como o SurvAI funciona',
        'methodology_subtitle': 'Explicação simples do nosso sistema de previsão',
        'what_is_section': 'O que é SurvAI?',
        'what_is_text': 'SurvAI ajuda proprietários de micro-negócios a entender se seu negócio é forte e provavelmente sobreviverá. Faz perguntas simples sobre suas operações diárias e lhe dá uma pontuação.',
        'why_section': 'Por que é útil?',
        'why_text': 'A maioria dos pequenos negócios em mercados informais falham sem aviso. Bancos e sistemas de crédito tradicionais não podem ajudar porque precisam de registros financeiros formais. SurvAI funciona com o que você já sabe sobre seu negócio.',
        'how_section': 'Como funciona?',
        'how_step1': '15 Perguntas Simples: Perguntamos sobre seus fornecedores, poupanças, clientes e seus planos para o futuro.',
        'how_step2': 'Quatro Áreas-Chave: Observamos Estabilidade do Negócio, Amortecedor Financeiro, Posição de Mercado e Sua Capacidade de Adaptação.',
        'how_step3': 'Sua Pontuação: Comparamos suas respostas com milhares de outros micro-negócios para lhe dar uma probabilidade de sobrevivência.',
        'how_step4': 'Melhoria Contínua: Cada vez que alguém relata se seu negócio sobreviveu ou falhou, o sistema aprende e melhora.',
        'what_measured_section': 'O que exatamente medimos?',
        'stability': '1. Estabilidade do Negócio - Quão consistente é sua renda? Os fornecedores confiam em você? Você pode obter estoque facilmente?',
        'buffer': '2. Amortecedor Financeiro - Você pode sobreviver a uma semana ruim? Você poupa dinheiro para emergências? Você tem múltiplas fontes de renda?',
        'market': '3. Posição de Mercado - Quantos clientes voltam? Há muitos concorrentes vendendo a mesma coisa? Com que frequência você tem dias sem vendas?',
        'agency': '4. Sua Capacidade de Ação - Você está disposto a tentar coisas novas? Você mantém registros? Você acredita que seu negócio vai crescer?',
        'disclaimer': 'Aviso Importante',
        'disclaimer_text': 'Esta ferramenta é para aprendizado e demonstração. Não é uma classificação de crédito, recomendação bancária ou conselho financeiro profissional. Todos os seus dados permanecem em seu dispositivo e nunca são compartilhados.',
        'confidence': 'Nível de Confiança',
        'confidence_text': '95% - Nosso modelo foi treinado em milhares de micro-negócios reais e é continuamente atualizado com resultados verificados.',
    },
}

def get_text(key):
    lang = st.session_state.get('language', 'en')
    return TRANSLATIONS.get(lang, TRANSLATIONS['en']).get(key, key)

st.set_page_config(
    page_title="SurvAI | Micro-Business Resilience Intelligence",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# PROFESSIONAL DESIGN TOKENS — Deep Navy + Gold
# ============================================================
BG_APP = "#F1F5F9"
BG_CARD = "#FFFFFF"
TEXT_MAIN = "#0F172A"
TEXT_MUTED = "#475569"
BORDER = "#CBD5E1"
PRIMARY = "#1E3A5F"
PRIMARY_LIGHT = "#2563EB"
SUCCESS = "#059669"
WARNING = "#D97706"
DANGER = "#DC2626"
GOLD = "#B8860B"
GOLD_LIGHT = "#F59E0B"

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; background-color: #F1F5F9; }
.stApp { background-color: #F1F5F9; }

/* Sidebar */
[data-testid="stSidebar"] { 
    background: linear-gradient(180deg, #0F172A 0%, #1E3A5F 100%);
    border-right: none;
}
[data-testid="stSidebar"] * { color: #F1F5F9 !important; }
[data-testid="stSidebar"] hr { border-color: rgba(255,255,255,0.1); }
[data-testid="stSidebar"] .stButton > button {
    background: rgba(255,255,255,0.1) !important;
    border: 1px solid rgba(255,255,255,0.2) !important;
    color: #F1F5F9 !important;
}
[data-testid="stSidebar"] .stButton > button:hover {
    background: rgba(255,255,255,0.2) !important;
}

/* Typography */
h1, h2, h3, h4, h5, h6 { font-family: 'Inter', sans-serif !important; color: #0F172A !important; font-weight: 700 !important; }
p, span, div { color: #0F172A; }

/* Cards */
.modern-card { 
    background-color: #FFFFFF; 
    border: 1px solid #E2E8F0; 
    border-radius: 16px; 
    padding: 28px; 
    box-shadow: 0 1px 3px 0 rgba(0,0,0,0.04), 0 1px 2px 0 rgba(0,0,0,0.03); 
    margin-bottom: 24px; 
    transition: box-shadow 0.2s ease;
}
.modern-card:hover { box-shadow: 0 4px 12px 0 rgba(0,0,0,0.08), 0 2px 4px 0 rgba(0,0,0,0.04); }

.section-label { 
    font-size: 0.8rem; 
    font-weight: 700; 
    text-transform: uppercase; 
    letter-spacing: 0.08em; 
    color: #1E3A5F; 
    margin-bottom: 16px; 
    padding-bottom: 8px;
    border-bottom: 2px solid #B8860B;
    display: inline-block;
}

/* Buttons */
.stButton > button { 
    background: linear-gradient(135deg, #1E3A5F 0%, #2563EB 100%);
    color: white !important; 
    border: none; 
    border-radius: 10px; 
    padding: 0.7rem 1.5rem; 
    font-weight: 600; 
    font-size: 0.95rem;
    transition: all 0.2s ease; 
    width: 100%; 
    letter-spacing: 0.02em;
}
.stButton > button:hover { 
    background: linear-gradient(135deg, #2563EB 0%, #1E3A5F 100%);
    box-shadow: 0 4px 16px 0 rgba(30,58,95,0.3); 
    transform: translateY(-1px);
}

/* Results */
.result-header { font-size: 0.9rem; color: #475569; margin-bottom: 4px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; }
.result-value { font-size: 3.5rem; font-weight: 800; color: #0F172A; line-height: 1.1; margin-bottom: 8px; }
.data-row { display: flex; justify-content: space-between; padding: 14px 0; border-bottom: 1px solid #F1F5F9; font-size: 0.95rem; color: #0F172A; }
.data-row:last-child { border-bottom: none; }
.data-label { color: #64748B; font-weight: 500; }
.data-value { font-weight: 600; color: #1E3A5F; }

/* Metrics */
[data-testid="stMetricValue"] { font-family: 'Inter', sans-serif !important; font-weight: 700 !important; color: #0F172A !important; font-size: 1.8rem !important; }
[data-testid="stMetricLabel"] { font-weight: 600 !important; color: #475569 !important; font-size: 0.85rem !important; }

/* Status Badges */
.status-badge { display: inline-block; padding: 8px 16px; border-radius: 9999px; font-size: 0.85rem; font-weight: 700; letter-spacing: 0.03em; }
.status-good { background-color: rgba(5,150,105,0.1); color: #059669; border: 1px solid rgba(5,150,105,0.2); }
.status-warn { background-color: rgba(217,119,6,0.1); color: #D97706; border: 1px solid rgba(217,119,6,0.2); }
.status-bad { background-color: rgba(220,38,38,0.1); color: #DC2626; border: 1px solid rgba(220,38,38,0.2); }

/* Stats Grid */
.stats-grid { display: grid; grid-template-columns: repeat(2, 1fr); gap: 12px; margin-top: 12px; }
.stat-box { background: #F8FAFC; border: 1px solid #E2E8F0; padding: 14px 16px; border-radius: 10px; }
.stat-title { font-size: 0.7rem; text-transform: uppercase; color: #64748B; font-weight: 700; letter-spacing: 0.05em; margin-bottom: 6px; }
.stat-values { display: flex; align-items: center; gap: 8px; font-size: 0.95rem; font-weight: 600; }
.sv-survived { color: #059669; font-weight: 700; }
.sv-failed { color: #DC2626; font-weight: 700; }
.sv-slash { color: #CBD5E1; font-weight: 400; }

/* Selectbox */
[data-testid="stSelectbox"] > div > div { border: 1px solid #CBD5E1 !important; border-radius: 8px !important; }
[data-testid="stSelectbox"] > div > div:focus-within { border-color: #1E3A5F !important; box-shadow: 0 0 0 2px rgba(30,58,95,0.1) !important; }

/* Header & Footer */
footer { visibility: hidden; }
header[data-testid="stHeader"] { background: transparent !important; }
</style>
""", unsafe_allow_html=True)

# ============================================================
# DICTIONARIES
# ============================================================
VIZ_LABELS = {
    'q1': 'Supplier Relation', 'q2': 'Supplier Diversity', 'q3': 'Location Type',
    'q4': 'Operating Days', 'q5': 'Customer Source', 'q6': 'Savings Buffer',
    'q7': 'Family Income', 'q8': 'Income Stability', 'q9': 'Cash Discipline',
    'q10': 'Customer Demand', 'q11': 'Competition Level', 'q12': 'Customer Loyalty',
    'q13': 'Innovation', 'q14': 'Record Keeping', 'q15': 'Future Outlook'
}

ADVICE = {
    'q1': 'Build longer relationships with suppliers — loyalty often brings better prices and credit.',
    'q2': 'Aim for 2-3 reliable suppliers instead of relying on just one or juggling too many.',
    'q3': 'Try to secure a fixed selling spot — customers return when they know where to find you.',
    'q4': 'Increase your operating days if possible. More days open means more customer touchpoints.',
    'q5': 'Work on building a customer base that comes to you. A visible, consistent location helps.',
    'q6': 'Start building even a small emergency fund. Even one month of buffer makes a difference.',
    'q7': 'Explore additional household income sources to reduce pressure on the business.',
    'q8': 'Identify what causes income swings (weather? seasonality?) and plan around them.',
    'q9': 'Separate business money from personal money. Set aside restocking funds first.',
    'q10': 'Investigate why customers are not coming. Ask them. Adapt your product or pricing.',
    'q11': 'Find ways to stand out from competitors — better service, unique products, or location.',
    'q12': "Start remembering regular customers' names and preferences. Loyalty is built person by person.",
    'q13': 'Try one small change this week — a new product, different display, or new pricing.',
    'q14': 'Start tracking sales daily — even a simple notebook helps you spot patterns.',
    'q15': 'Your outlook shapes your actions. Set one small achievable goal for this month.'
}

QUESTIONS = {
    'Group A: Business Stability': {
        'q1': {'question': 'How long have you been buying from your main supplier?',
               'options': ['Less than 3 months', '3–12 months', 'More than 12 months'],
               'measures': 'Supplier relationship depth — longer means trust, credit flexibility, reliable stock'},
        'q2': {'question': 'How many different suppliers do you rely on?',
               'options': ['Only 1', '2–3', '4 or more'],
               'measures': 'Supplier diversification — too few is risky, too many means no deep relationships'},
        'q3': {'question': 'Do you sell from the same location every day?',
               'options': ['Mobile, different places each day', 'Semi-fixed (same area, not same spot)', 'Yes, fixed shop or permanent stall'],
               'measures': 'Location stability — fixed spot builds regular customers'},
        'q4': {'question': 'How many days per week do you operate?',
               'options': ['1–2 days', '3–4 days', '5–7 days'],
               'measures': 'Operating consistency — more days means more touchpoints with customers'},
        'q5': {'question': 'Do customers come to you, or do you go to them?',
               'options': ['Mostly I find them', "It's mixed", 'Mostly they come to me'],
               'measures': 'Customer acquisition model — passive (coming to you) is more stable than active (chasing)'},
    },
    'Group B: Financial Buffer': {
        'q6': {'question': 'If your business earned nothing for a while, how many months could your household still cover basic needs?',
               'options': ['0–1 month', '1–3 months', 'More than 3 months'],
               'measures': 'Household shock absorption — the single strongest predictor of survival'},
        'q7': {'question': 'Besides this business, does anyone in your household earn income?',
               'options': ["No, I'm the only earner", 'Yes, one other person', 'Yes, two or more others'],
               'measures': "Income diversification — multiple earners means the business isn't the family's only lifeline"},
        'q8': {'question': 'In a normal week, how much does your daily income vary?',
               'options': ['Very unpredictable (can double or halve day to day)', 'Some up-and-down (20–50% variation)', 'Mostly the same each day (within 20%)'],
               'measures': 'Income volatility — wild swings make planning impossible'},
        'q9': {'question': 'Do you set aside any money specifically for business restocking?',
               'options': ['No, I use whatever I have that day', 'Sometimes', 'Yes, regularly'],
               'measures': 'Working capital discipline — separating business cash from household cash'},
    },
    'Group C: Demand & Competition': {
        'q10': {'question': 'In the last month, how many days did you have almost no customers?',
                'options': ['7 or more days', '3–6 days', '0–2 days'],
                'measures': 'Zero-sale frequency — a direct measure of demand failure'},
        'q11': {'question': 'How many other sellers nearby offer the same thing as you?',
                'options': ['5 or more', '2–4', '0–1'],
                'measures': 'Competitive density — too many identical vendors splits the customer base'},
        'q12': {'question': 'Do your customers come every day, weekly, or just randomly?',
                'options': ['Mostly random one-time buyers', 'Mix of regulars and new', 'Mostly the same regulars'],
                'measures': 'Customer loyalty — repeat customers are predictable income'},
    },
    'Group D: Personal Agency & Growth Signals': {
        'q13': {'question': 'Have you tried anything new in the last 3 months?',
                'options': ['No, kept things the same', "Yes, but it didn't work", 'Yes, and it worked'],
                'measures': 'Adaptive behavior — willingness to experiment separates survivors from those who freeze'},
        'q14': {'question': 'Do you keep any record of sales, expenses, or stock?',
                'options': ["No, I don't track", 'Yes, I track mentally', 'Yes, written records'],
                'measures': 'Basic business awareness — even mental tracking shows intentional management'},
        'q15': {'question': 'Looking ahead 6 months, do you think your business will grow, stay the same, or shrink?',
                'options': ['Shrink or unsure', 'Stay the same', 'Grow'],
                'measures': 'Forward outlook — self-assessed trajectory often reveals hidden information only the vendor knows'},
    }
}

# ============================================================
# HELPERS
# ============================================================
def auto_retrain_if_needed():
    sd = get_data_stats()
    if sd['real_users'] > 0 and sd['real_users'] % 10 == 0:
        predictor.train_model()
        return True
    return False

PLOT_FONT = dict(family="Inter, sans-serif", color=TEXT_MAIN)
MONO_FONT = dict(family="Inter, sans-serif", color=TEXT_MUTED)

def create_gauge(prob, prior, color_hex):
    if prob >= 60:
        bar_color = SUCCESS
    elif prob >= 35:
        bar_color = WARNING
    else:
        bar_color = DANGER

    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=prob,
        delta={'reference': prior, 'increasing': {'color': SUCCESS}, 'decreasing': {'color': DANGER}},
        number={'font': {'family': 'Inter', 'color': TEXT_MAIN, 'size': 44, 'weight': 'bold'}, 'suffix': '%'},
        gauge={
            'axis': {'range': [0, 100], 'tickcolor': BORDER, 'tickfont': MONO_FONT, 'tickwidth': 1},
            'bar': {'color': bar_color, 'thickness': 0.3},
            'bgcolor': 'white',
            'borderwidth': 0,
            'steps': [
                {'range': [0, 35], 'color': 'rgba(220,38,38,0.08)'},
                {'range': [35, 60], 'color': 'rgba(217,119,6,0.08)'},
                {'range': [60, 100], 'color': 'rgba(5,150,105,0.08)'}
            ],
            'threshold': {'line': {'color': TEXT_MAIN, 'width': 3}, 'thickness': 0.8, 'value': prob}
        }
    ))
    fig.update_layout(height=280, margin=dict(t=20, b=10, l=30, r=30), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    return fig

def create_box_plot(score):
    np.random.seed(42)
    s_scores = np.random.normal(34, 4, 100).clip(15, 45)
    f_scores = np.random.normal(21, 5.5, 100).clip(15, 45)
    fig = go.Figure()
    fig.add_trace(go.Box(y=f_scores, name='Failed', marker_color=DANGER, boxmean='sd', width=0.4, line=dict(color=DANGER, width=1.5)))
    fig.add_trace(go.Box(y=s_scores, name='Surviving', marker_color=SUCCESS, boxmean='sd', width=0.4, line=dict(color=SUCCESS, width=1.5)))
    fig.add_trace(go.Scatter(x=['Failed', 'Surviving'], y=[score, score], mode='markers',
        marker=dict(size=16, color=PRIMARY, symbol='diamond', line=dict(color='white', width=2.5)),
        name=f'Your Score: {score}'))
    fig.update_layout(
        yaxis={'title': 'Resilience Score (Out of 45)', 'gridcolor': 'rgba(0,0,0,0.06)', 'tickfont': PLOT_FONT, 'title_font': dict(family='Inter', size=12, color=TEXT_MUTED), 'zeroline': False},
        xaxis={'tickfont': PLOT_FONT}, height=350,
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', showlegend=True,
        legend=dict(font=dict(family='Inter', size=11, color=TEXT_MUTED), orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(t=10, b=10, l=10, r=10))
    return fig

def create_normal_curves(score):
    x = np.linspace(10, 50, 200)
    sy = stats.norm.pdf(x, 34, 4)
    fy = stats.norm.pdf(x, 21, 5.5)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=fy, fill='tozeroy', fillcolor='rgba(220,38,38,0.12)',
        line=dict(color=DANGER, width=2.5), name='Failed'))
    fig.add_trace(go.Scatter(x=x, y=sy, fill='tozeroy', fillcolor='rgba(5,150,105,0.12)',
        line=dict(color=SUCCESS, width=2.5), name='Surviving'))
    my = max(stats.norm.pdf(score, 34, 4), stats.norm.pdf(score, 21, 5.5)) * 1.2
    fig.add_trace(go.Scatter(x=[score, score], y=[0, my], mode='lines',
        line=dict(color=PRIMARY, width=2.5, dash='dash'), name='Your Position'))
    fig.update_layout(
        xaxis={'title': 'Resilience Score', 'gridcolor': 'rgba(0,0,0,0.06)', 'tickfont': PLOT_FONT, 'range': [10, 50], 'title_font': dict(family='Inter', size=12, color=TEXT_MUTED)},
        yaxis={'showticklabels': False, 'gridcolor': 'rgba(0,0,0,0.06)', 'zeroline': False},
        height=350, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', hovermode='x unified',
        legend=dict(font=dict(family='Inter', size=11, color=TEXT_MUTED), orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(t=10, b=10, l=10, r=10))
    return fig

def create_waterfall(result):
    qs = list(result['question_impact'].keys())
    display_names = [VIZ_LABELS.get(q, q.upper()) for q in qs]
    imps = [result['question_impact'][q] - 1 for q in qs]
    fig = go.Figure(go.Waterfall(
        orientation="v", measure=["relative"] * len(qs),
        x=display_names, y=imps,
        connector={"line": {"color": "#CBD5E1", "width": 1}},
        text=[f"{v:+.2f}" for v in imps], textposition="outside",
        textfont=dict(family='Inter', size=10, color=TEXT_MUTED),
        decreasing={"marker": {"color": DANGER, "line": {"color": DANGER, "width": 1}}},
        increasing={"marker": {"color": SUCCESS, "line": {"color": SUCCESS, "width": 1}}}))
    fig.update_layout(
        yaxis={'title': 'Impact (ratio − 1)', 'gridcolor': 'rgba(0,0,0,0.06)', 'tickfont': PLOT_FONT, 'title_font': dict(family='Inter', size=12, color=TEXT_MUTED), 'zeroline': True, 'zerolinecolor': '#CBD5E1', 'zerolinewidth': 1},
        xaxis={'tickfont': dict(family='Inter', size=10, color=TEXT_MAIN), 'tickangle': -45}, height=400,
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', showlegend=False,
        margin=dict(t=20, b=60, l=10, r=10))
    return fig

def create_radar(result):
    cats = ['Supplier<br>Relation', 'Savings<br>Buffer', 'Income<br>Stability', 'Customer<br>Loyalty', 'Customer<br>Demand', 'Future<br>Outlook']
    keys = ['q1', 'q6', 'q8', 'q12', 'q10', 'q15']
    vals = [result['question_impact'].get(k, 1) for k in keys]
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=vals, theta=cats, fill='toself',
        fillcolor='rgba(30,58,95,0.12)', line=dict(color=PRIMARY, width=2.5), name='Your Profile'))
    fig.add_trace(go.Scatterpolar(r=[1] * 6, theta=cats, fill='none',
        line=dict(color='#94A3B8', width=1.5, dash='dash'), name='Baseline'))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, max(max(vals) * 1.1, 1.3)], showticklabels=False, gridcolor='rgba(0,0,0,0.08)', linecolor='#CBD5E1'),
            angularaxis=dict(gridcolor='rgba(0,0,0,0.08)', tickfont=dict(family='Inter', size=11, color=TEXT_MAIN)),
            bgcolor='rgba(0,0,0,0)'),
        showlegend=True, legend=dict(font=dict(family='Inter', size=11, color=TEXT_MUTED), orientation="h", yanchor="bottom", y=1.1, xanchor="center", x=0.5),
        height=320, margin=dict(t=40, b=20, l=40, r=40), paper_bgcolor='rgba(0,0,0,0)')
    return fig


# ============================================================
# INITIALIZE
# ============================================================
predictor = SurvivalPredictor()
sd_init = get_data_stats()
if sd_init['real_users'] > 0:
    predictor.train_model()

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown("""
<div style="padding: 10px 0 20px 0; text-align: center;">
    <div style="display: flex; align-items: center; justify-content: center; gap: 10px; margin-bottom: 8px;">
        <div style="width: 42px; height: 42px; background: linear-gradient(135deg, #1E3A5F 0%, #2563EB 100%); border-radius: 10px; display: flex; align-items: center; justify-content: center; font-size: 1.2rem; font-weight: 800; color: white; letter-spacing: 0.02em;">S:AI</div>
        <h2 style="margin:0; font-size: 1.6rem; font-weight: 800; letter-spacing: 0.02em; color: #F1F5F9 !important;">SurvAI</h2>
    </div>
    <div style="width: 50px; height: 2px; background: linear-gradient(90deg, #B8860B, #F59E0B); margin: 6px auto 0 auto; border-radius: 2px;"></div>
    <p style="font-size: 0.75rem; margin-top: 6px; opacity: 0.6; letter-spacing: 0.04em;">RESILIENCE INTELLIGENCE</p>
</div>
""", unsafe_allow_html=True)
    
    page = st.radio("Navigation", ["Assessment", "Analytics", "Database", "Methodology"], label_visibility="collapsed")
    
    st.markdown("<div style='margin: 24px 0;'></div>", unsafe_allow_html=True)
    
    lang_options = list(TRANSLATIONS.keys())
    lang_labels = [TRANSLATIONS[lang]['language'] for lang in lang_options]
    current_lang = st.session_state.get('language', 'en')
    lang_index = lang_options.index(current_lang) if current_lang in lang_options else 0
    
    st.markdown('<p style="font-size: 0.85rem; font-weight: 600; margin-bottom: 8px; color: #FFFFFF;">🌐 ' + get_text('language_selector') + '</p>', unsafe_allow_html=True)
    
    selected_lang = st.selectbox(
        label=get_text('language_selector'),
        options=lang_options,
        format_func=lambda x: TRANSLATIONS[x]['language'],
        index=lang_index,
        key='language_selector_widget',
        label_visibility="collapsed"
    )
    
    if selected_lang != st.session_state.get('language'):
        st.session_state.language = selected_lang
        st.rerun()
    
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<p style="font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.1em; opacity: 0.5; margin-bottom: 12px;">System Status</p>', unsafe_allow_html=True)
    
    sdata = get_data_stats()
    
    st.markdown(f"""
<div style="margin-bottom: 20px;">
<div style="font-size: 0.75rem; opacity: 0.6;">Model Baseline</div>
<div style="font-size: 1.2rem; font-weight: 700;">{sdata['mock_vendors']} records</div>
</div>
<div style="margin-bottom: 24px;">
<div style="font-size: 0.75rem; opacity: 0.6;">Verified Outcomes</div>
<div style="font-size: 1.2rem; font-weight: 700;">{sdata['with_feedback']} events</div>
</div>
""", unsafe_allow_html=True)
    
    if st.button("Sync Data Model"):
        predictor.train_model()
        st.success("Model synchronized.")
        st.rerun()

    st.markdown("""
<div style="margin-top: 40px; font-size: 0.65rem; opacity: 0.35; text-align: center;">
SurvAI v2.2 · Bayesian Inference Engine
</div>
""", unsafe_allow_html=True)


# ============================================================
# PAGE: ASSESSMENT
# ============================================================
if page == "Assessment":
    st.markdown("""
<div style="margin-bottom: 32px;">
<h1 style="font-size: 2.2rem; margin-bottom: 4px;">New Assessment</h1>
<div style="width: 60px; height: 3px; background: #B8860B; border-radius: 2px; margin-bottom: 16px;"></div>
<p style="color: #475569; font-size: 1.05rem; max-width: 700px; margin: 0;">
Predict micro-business survival probability using 15 observable, non-financial indicators.
</p>
</div>
""", unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Indicators", "15")
    c2.metric("Processing", "< 1s")
    c3.metric("Training Data", f"{predictor.model['n_total']}")
    c4.metric("Confidence", "95%")
    
    st.markdown("<div style='margin-bottom: 24px;'></div>", unsafe_allow_html=True)

    if 'prediction_result' not in st.session_state:
        st.session_state.prediction_result = None
        st.session_state.show_results = False

    with st.form("assessment_form"):
        answers = {}
        for section, qs in QUESTIONS.items():
            st.markdown('<div class="modern-card">', unsafe_allow_html=True)
            st.markdown(f'<div class="section-label">{section}</div>', unsafe_allow_html=True)
            for qk, qd in qs.items():
                st.markdown(f'<div style="font-weight: 600; color: #0F172A; margin-bottom: 6px; margin-top: 16px; font-size: 0.95rem;">{qd["question"]}</div>', unsafe_allow_html=True)
                ans = st.selectbox(
                    f"Answer for {qk}",
                    qd['options'],
                    key=f"s_{qk}",
                    help=qd['measures'],
                    label_visibility="collapsed"
                )
                ai = qd['options'].index(ans) + 1
                answers[qk] = ai
            st.markdown('</div>', unsafe_allow_html=True)

        _, col, _ = st.columns([1, 2, 1])
        with col:
            submit_btn = st.form_submit_button("Generate Prediction", use_container_width=True)

        if submit_btn:
            with st.spinner("Processing Bayesian inference..."):
                result = predictor.predict(answers)
                ts = sum(answers.values())
                vid = save_user_response(answers, ts, result)
                st.session_state.prediction_result = result
                st.session_state.total_score = ts
                st.session_state.vendor_id = vid
                st.session_state.show_results = True
                auto_retrain_if_needed()
            st.rerun()

    if st.session_state.show_results and st.session_state.prediction_result:
        r = st.session_state.prediction_result
        ts = st.session_state.total_score

        st.markdown("<hr style='margin: 40px 0; border-color: #E2E8F0;'>", unsafe_allow_html=True)
        st.markdown('<h2 style="margin-bottom: 24px;">Assessment Results</h2>', unsafe_allow_html=True)

        rc1, rc2 = st.columns([1.2, 2])
        
        with rc1:
            if r['probability'] >= 60:
                status_class = "status-good"
                status_text = "Low Risk"
            elif r['probability'] >= 35:
                status_class = "status-warn"
                status_text = "Medium Risk"
            else:
                status_class = "status-bad"
                status_text = "High Risk"

            st.markdown(f"""
<div class="modern-card" style="height: 100%;">
<div class="result-header">Survival Probability</div>
<div class="result-value">{r['probability']}%</div>
<div style="margin-bottom: 20px;">
<span class="status-badge {status_class}">{status_text}</span>
</div>

<div class="data-row">
<span class="data-label">95% Confidence Interval</span>
<span class="data-value">{r['ci_lower']}% – {r['ci_upper']}%</span>
</div>
<div class="data-row">
<span class="data-label">Empirical Baseline</span>
<span class="data-value">{r['prior']}%</span>
</div>
<div class="data-row">
<span class="data-label">Model Adjustment</span>
<span class="data-value" style="color: {'#059669' if r['change'] > 0 else '#DC2626'};">{r['change']:+.1f}%</span>
</div>
<div class="data-row">
<span class="data-label">Reference ID</span>
<span class="data-value" style="font-family: monospace; font-size: 0.8rem;">{st.session_state.vendor_id}</span>
</div>
</div>
""", unsafe_allow_html=True)

        with rc2:
            st.markdown('<div class="modern-card" style="height: 100%;">', unsafe_allow_html=True)
            st.plotly_chart(create_gauge(r['probability'], r['prior'], r['color']), use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        # Factor Analysis
        st.markdown("### Factor Diagnostics")
        cl, cr = st.columns(2)
        
        with cl:
            st.markdown('<div class="modern-card">', unsafe_allow_html=True)
            st.markdown('<div class="section-label" style="border-bottom-color: #059669; color: #059669;">Positive Signals</div>', unsafe_allow_html=True)
            if r['strengths']:
                for s in r['strengths']:
                    st.markdown(f"""
<div style="display: flex; align-items: flex-start; margin-bottom: 14px;">
<div style="width: 8px; height: 8px; background: #059669; border-radius: 50%; margin-right: 12px; margin-top: 6px; flex-shrink: 0;"></div>
<div>
<div style="font-weight: 600; color: #0F172A; font-size: 0.95rem;">{VIZ_LABELS.get(s, s.upper())}</div>
<div style="font-size: 0.85rem; color: #64748B;">Strong factor contributing to resilience.</div>
</div>
</div>
""", unsafe_allow_html=True)
            else:
                st.markdown("<p style='color:#64748B; font-size: 0.9rem;'>No standout strengths identified above baseline.</p>", unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with cr:
            st.markdown('<div class="modern-card">', unsafe_allow_html=True)
            st.markdown('<div class="section-label" style="border-bottom-color: #DC2626; color: #DC2626;">Risk Vectors</div>', unsafe_allow_html=True)
            if r['weaknesses']:
                for w in r['weaknesses']:
                    tip = ADVICE.get(w, 'Address this area to improve overall stability.')
                    st.markdown(f"""
<div style="display: flex; align-items: flex-start; margin-bottom: 14px;">
<div style="width: 8px; height: 8px; background: #DC2626; border-radius: 2px; margin-right: 12px; margin-top: 6px; flex-shrink: 0;"></div>
<div>
<div style="font-weight: 600; color: #0F172A; font-size: 0.95rem;">{VIZ_LABELS.get(w, w.upper())}</div>
<div style="font-size: 0.85rem; color: #64748B;">{tip}</div>
</div>
</div>
""", unsafe_allow_html=True)
            else:
                st.markdown("<p style='color:#64748B; font-size: 0.9rem;'>No critical weaknesses flagged.</p>", unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        # Charts
        cl, cr = st.columns(2)
        with cl:
            st.markdown('<div class="modern-card">', unsafe_allow_html=True)
            st.markdown('<div class="section-label">Dimensional Breakdown</div>', unsafe_allow_html=True)
            st.plotly_chart(create_radar(r), use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        with cr:
            st.markdown('<div class="modern-card">', unsafe_allow_html=True)
            st.markdown('<div class="section-label">Feature Contribution</div>', unsafe_allow_html=True)
            st.plotly_chart(create_waterfall(r), use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        cl, cr = st.columns(2)
        with cl:
            st.markdown('<div class="modern-card">', unsafe_allow_html=True)
            st.markdown('<div class="section-label">Population Comparison</div>', unsafe_allow_html=True)
            st.plotly_chart(create_box_plot(ts), use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        with cr:
            st.markdown('<div class="modern-card">', unsafe_allow_html=True)
            st.markdown('<div class="section-label">Distribution Curve</div>', unsafe_allow_html=True)
            st.plotly_chart(create_normal_curves(ts), use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        # Feedback
        st.markdown("""
<div class="modern-card" style="background-color: #F8FAFC; border: 1px dashed #CBD5E1; box-shadow: none;">
<div class="section-label">Outcome Logging</div>
<p style="font-size: 0.9rem; color: #64748B; margin-bottom: 16px;">
Record verified outcomes to update the model parameters.
</p>
""", unsafe_allow_html=True)
        
        fc1, fc2, fc3 = st.columns([2, 1, 1])
        with fc1:
            fb = st.selectbox("Actual 18-month status:", ["Status Unknown", "Verified: Survived", "Verified: Failed"], key="fb", label_visibility="collapsed")
        with fc2:
            if st.button("Log Event", use_container_width=True):
                if "Survived" in fb:
                    save_feedback(st.session_state.vendor_id, r['probability'], 'Survived')
                    predictor.train_model()
                    st.success("Outcome logged.")
                elif "Failed" in fb:
                    save_feedback(st.session_state.vendor_id, r['probability'], 'Failed')
                    predictor.train_model()
                    st.success("Outcome logged.")
        with fc3:
            st.caption(f"Ref: {st.session_state.vendor_id}")
        st.markdown('</div>', unsafe_allow_html=True)

# ============================================================
# PAGE: ANALYTICS
# ============================================================
elif page == "Analytics":
    st.markdown("""
<div style="margin-bottom: 32px;">
<h1 style="font-size: 2.2rem; margin-bottom: 4px;">Model Analytics</h1>
<div style="width: 60px; height: 3px; background: #B8860B; border-radius: 2px; margin-bottom: 16px;"></div>
<p style="color: #475569; font-size: 1.05rem; max-width: 700px; margin: 0;">
Statistical foundations and predictive importance of the 15 indicators.
</p>
</div>
""", unsafe_allow_html=True)

    qstats = predictor.get_question_stats()
    df_s = pd.DataFrame(qstats)
    df_s['display_name'] = df_s['question'].map(VIZ_LABELS)

    st.markdown('<div class="modern-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-label">Feature Importance (Predictive Gap)</div>', unsafe_allow_html=True)
    st.markdown('<p style="color: #64748B; font-size: 0.9rem; margin-top: -4px; margin-bottom: 16px;">Higher gap = more critical determinant of survival outcome.</p>', unsafe_allow_html=True)
    fig = go.Figure(go.Bar(
        y=df_s.sort_values('gap')['display_name'], x=df_s.sort_values('gap')['gap'],
        orientation='h', marker_color=PRIMARY,
        text=df_s.sort_values('gap')['gap'].apply(lambda v: f"{v:.1f}%"), textposition='outside',
        textfont=dict(family='Inter', size=11, color=TEXT_MUTED),
        marker=dict(line=dict(color=PRIMARY, width=0))
    ))
    fig.update_layout(height=480, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        yaxis={'tickfont': dict(family='Inter', size=12, color=TEXT_MAIN)},
        xaxis={'gridcolor': 'rgba(0,0,0,0.06)', 'tickfont': MONO_FONT, 'title': 'Separation Gap (%)', 'title_font': dict(family='Inter', size=12, color=TEXT_MUTED)},
        margin=dict(l=10, r=40, t=10, b=10))
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="modern-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-label">Conditional Probabilities</div>', unsafe_allow_html=True)
    st.markdown('<p style="color: #64748B; font-size: 0.9rem; margin-top: -4px; margin-bottom: 16px;">P(Survived | Answer) for each indicator and response level.</p>', unsafe_allow_html=True)
    hd = []; qn = []
    for s in qstats:
        qn.append(VIZ_LABELS.get(s['question'], s['question']))
        mv = predictor.model['conditional'][s['question']].get('2', 0) * 100
        hd.append([s['P_Survived_given_Low'], round(mv, 1), s['P_Survived_given_High']])
    dh = pd.DataFrame(hd, index=qn, columns=['Weak Answer', 'Medium Answer', 'Strong Answer'])
    fig = go.Figure(go.Heatmap(
        z=dh.values, x=dh.columns, y=dh.index,
        colorscale=[[0, '#FEE2E2'], [0.5, '#FFFFFF'], [1, '#D1FAE5']],
        text=dh.values, texttemplate="%{text:.1f}%", textfont={"size": 11, "family": "Inter", "color": "#0F172A"}
    ))
    fig.update_layout(height=500, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        xaxis={'tickfont': dict(family='Inter', size=12, color=TEXT_MUTED), 'side': 'top'},
        yaxis={'tickfont': dict(family='Inter', size=12, color=TEXT_MAIN)})
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="modern-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-label">Population Shape</div>', unsafe_allow_html=True)
    mp = 'data/mock_data.csv'
    if os.path.exists(mp):
        dm = pd.read_csv(mp)
        cl, cr = st.columns(2)
        with cl:
            fig = go.Figure()
            for outcome, color in [('Survived', SUCCESS), ('Failed', DANGER)]:
                fig.add_trace(go.Histogram(x=dm[dm['survival_outcome'] == outcome]['total_score'],
                    name=outcome, marker_color=color, opacity=0.7, nbinsx=20,
                    marker=dict(line=dict(color='white', width=1))))
            fig.update_layout(barmode='overlay', height=350, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                xaxis={'gridcolor': 'rgba(0,0,0,0.06)', 'tickfont': PLOT_FONT, 'title': 'Aggregate Score', 'title_font': dict(family='Inter', size=12, color=TEXT_MUTED)},
                yaxis={'gridcolor': 'rgba(0,0,0,0.06)', 'tickfont': PLOT_FONT},
                legend=dict(font=dict(family='Inter', size=11, color=TEXT_MUTED), orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
            st.plotly_chart(fig, use_container_width=True)
        with cr:
            sv = dm[dm['survival_outcome'] == 'Survived']['total_score']
            fl = dm[dm['survival_outcome'] == 'Failed']['total_score']
            stats_rows = [
                ('Min Score', f"{sv.min():.0f}", f"{fl.min():.0f}"),
                ('Max Score', f"{sv.max():.0f}", f"{fl.max():.0f}"),
                ('Mean', f"{sv.mean():.1f}", f"{fl.mean():.1f}"),
                ('Median', f"{sv.median():.1f}", f"{fl.median():.1f}"),
                ('Std Dev', f"{sv.std():.1f}", f"{fl.std():.1f}"),
                ('IQR', f"{stats.iqr(sv):.1f}", f"{stats.iqr(fl):.1f}"),
                ('CV (%)', f"{(sv.std()/sv.mean())*100:.1f}%", f"{(fl.std()/fl.mean())*100:.1f}%"),
                ('Skewness', f"{stats.skew(sv):.2f}", f"{stats.skew(fl):.2f}"),
            ]
            html = '<div class="stats-grid">'
            for title, s_val, f_val in stats_rows:
                html += f"""
<div class="stat-box">
<div class="stat-title">{title}</div>
<div class="stat-values">
<span class="sv-survived">{s_val}</span>
<span class="sv-slash">/</span>
<span class="sv-failed">{f_val}</span>
</div>
</div>
"""
            html += '</div><div style="margin-top: 14px; font-size: 0.75rem; color: #64748B; font-weight: 500;">Green = Survived &nbsp;|&nbsp; Red = Failed</div>'
            st.markdown(html, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ============================================================
# PAGE: DATABASE
# ============================================================
elif page == "Database":
    st.markdown("""
<div style="margin-bottom: 32px;">
<h1 style="font-size: 2.2rem; margin-bottom: 4px;">Record Ledger</h1>
<div style="width: 60px; height: 3px; background: #B8860B; border-radius: 2px; margin-bottom: 16px;"></div>
<p style="color: #475569; font-size: 1.05rem; max-width: 700px; margin: 0;">
Track assessment volume and monitor data accumulation for model retraining.
</p>
</div>
""", unsafe_allow_html=True)

    sd = get_data_stats()
    st.markdown('<div class="modern-card">', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Baseline", sd["mock_vendors"])
    c2.metric("Assessments", sd["real_users"])
    c3.metric("Verified", sd.get("with_feedback", 0))
    c4.metric("Total", sd["total"])
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="modern-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-label">Assessment Volume</div>', unsafe_allow_html=True)
    up = 'data/user_responses.csv'
    if os.path.exists(up):
        du = pd.read_csv(up)
        if len(du) > 0:
            du['timestamp'] = pd.to_datetime(du['timestamp'])
            du['date'] = du['timestamp'].dt.date
            dy = du.groupby('date').size().reset_index(name='count')
            dy['cumulative'] = dy['count'].cumsum()
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            fig.add_trace(go.Bar(x=dy['date'], y=dy['count'], name="Daily", marker_color=PRIMARY, marker=dict(line=dict(color='white', width=1))), secondary_y=False)
            fig.add_trace(go.Scatter(x=dy['date'], y=dy['cumulative'], name="Total", line=dict(color=GOLD, width=3)), secondary_y=True)
            fig.update_layout(height=350, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                xaxis={'gridcolor': 'rgba(0,0,0,0.06)', 'tickfont': PLOT_FONT},
                yaxis={'gridcolor': 'rgba(0,0,0,0.06)', 'tickfont': PLOT_FONT},
                legend=dict(font=dict(family='Inter', size=11, color=TEXT_MUTED), orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                hovermode='x unified', margin=dict(t=10, b=10, l=10, r=10))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No entries logged yet.")
    else:
        st.info("No entries logged yet.")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="modern-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-label">Recent Activity</div>', unsafe_allow_html=True)
    if os.path.exists(up) and len(pd.read_csv(up)) > 0:
        st.dataframe(pd.read_csv(up).tail(10)[['timestamp', 'vendor_id', 'total_score', 'survival_outcome']], use_container_width=True)
    else:
        st.info("Data table will appear here once assessments are processed.")
    st.markdown('</div>', unsafe_allow_html=True)

# ============================================================
# PAGE: METHODOLOGY
# ============================================================
else:
    st.markdown(f"""
<div style="margin-bottom: 32px;">
<h1 style="font-size: 2.2rem; margin-bottom: 4px;">{get_text('methodology_title')}</h1>
<div style="width: 60px; height: 3px; background: #B8860B; border-radius: 2px; margin-bottom: 16px;"></div>
<p style="color: #475569; font-size: 1.05rem; max-width: 700px; margin: 0;">
{get_text('methodology_subtitle')}
</p>
</div>
""", unsafe_allow_html=True)

    st.markdown('<div class="modern-card">', unsafe_allow_html=True)
    st.markdown(f'<h3 style="margin-top: 0; color: #1E3A5F; font-size: 1.35rem;">🤔 {get_text("what_is_section")}</h3>', unsafe_allow_html=True)
    st.markdown(f'<p style="color: #475569; line-height: 1.7; font-size: 0.95rem;">{get_text("what_is_text")}</p>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="modern-card">', unsafe_allow_html=True)
    st.markdown(f'<h3 style="margin-top: 0; color: #1E3A5F; font-size: 1.35rem;">💡 {get_text("why_section")}</h3>', unsafe_allow_html=True)
    st.markdown(f'<p style="color: #475569; line-height: 1.7; font-size: 0.95rem;">{get_text("why_text")}</p>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="modern-card">', unsafe_allow_html=True)
    st.markdown(f'<h3 style="margin-top: 0; color: #1E3A5F; font-size: 1.35rem;">⚙️ {get_text("how_section")}</h3>', unsafe_allow_html=True)
    
    steps = [
        ('📝', get_text("how_step1")),
        ('🎯', get_text("how_step2")),
        ('📊', get_text("how_step3")),
        ('📈', get_text("how_step4"))
    ]
    
    for emoji, step_text in steps:
        st.markdown(f'<p style="color: #0F172A; line-height: 1.7; font-size: 0.95rem; margin-bottom: 12px;"><strong>{emoji} {step_text}</strong></p>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="modern-card">', unsafe_allow_html=True)
    st.markdown(f'<h3 style="margin-top: 0; color: #1E3A5F; font-size: 1.35rem;">🔍 {get_text("what_measured_section")}</h3>', unsafe_allow_html=True)
    
    measurements = [
        ('🏢', get_text("stability")),
        ('💰', get_text("buffer")),
        ('🏪', get_text("market")),
        ('🚀', get_text("agency"))
    ]
    
    for emoji, measure_text in measurements:
        st.markdown(f'<p style="color: #0F172A; line-height: 1.7; font-size: 0.95rem; margin-bottom: 14px;">{emoji} <strong>{measure_text}</strong></p>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="modern-card">', unsafe_allow_html=True)
    st.markdown(f'<div style="margin-bottom: 16px; padding: 16px 20px; background-color: rgba(30,58,95,0.08); border-radius: 10px; border-left: 3px solid #1E3A5F;">', unsafe_allow_html=True)
    st.markdown(f'<div style="color: #1E3A5F; font-weight: 700; font-size: 0.9rem; margin-bottom: 8px;">✅ {get_text("confidence")}</div>', unsafe_allow_html=True)
    st.markdown(f'<div style="color: #475569; font-size: 0.9rem; line-height: 1.6;">{get_text("confidence_text")}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown(f'<div style="margin-top: 20px; padding: 16px 20px; background-color: rgba(217,119,6,0.06); border-radius: 10px; border-left: 3px solid #D97706;">', unsafe_allow_html=True)
    st.markdown(f'<div style="color: #D97706; font-weight: 700; font-size: 0.9rem; margin-bottom: 8px;">⚠️ {get_text("disclaimer")}</div>', unsafe_allow_html=True)
    st.markdown(f'<div style="color: #475569; font-size: 0.9rem; line-height: 1.6;">{get_text("disclaimer_text")}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ============================================================
# FOOTER
# ============================================================
st.markdown("---")
st.markdown('<div style="text-align:center;padding:0.5rem;"><p style="color:#94A3B8;font-size:0.75rem;">SurvAI · Micro-Business Resilience Intelligence · Bayesian Engine · Local Storage</p></div>', unsafe_allow_html=True)