import "./About.css";

/* ── SAMPLE DATA — replace with your real info ── */

const MECHANISM = [
  {
    icon: "🧹",
    title: "Text Preprocessing",
    description:
      "Raw news articles are cleaned by removing HTML tags, special characters, and stopwords, then lemmatized to normalize the text for model input.",
    extra: null,
  },
  {
    icon: "📊",
    title: "Feature Extraction",
    description:
      "TF-IDF vectorization converts cleaned text into numerical feature vectors that machine learning models can process efficiently.",
    extra: null,
  },
  {
    icon: "🤖",
    title: "ML Models",
    description:
      "Multiple models are trained for prediction. The Hybrid model averages probabilities across all available models for the best result.",
    extra: "models",
  },
  {
    icon: "🔄",
    title: "Prediction Pipeline",
    description:
      "Each article passes through the full pipeline — from raw input to a final FAKE or REAL verdict with a confidence score.",
    extra: "pipeline",
  },
];

const MODELS_LIST = [
  { label: "Naive Bayes",         highlight: false },
  { label: "SVM",                 highlight: false },
  { label: "Logistic Regression", highlight: false },
  { label: "BiLSTM + CNN",        highlight: false },
  { label: "BERT",                highlight: false },
  { label: "Hybrid (Best)",       highlight: true  },
];

const PIPELINE_STEPS = [
  "Raw Article",
  "Clean Text",
  "TF-IDF Vectors",
  "Model Predict",
  "Avg Probability",
  "FAKE / REAL",
];

const TEAM = [
  { name: "Aarav Sharma",   role: "Team Lead & ML Engineer"  },
  { name: "Priya Patel",    role: "NLP & Preprocessing"      },
  { name: "Rohan Mehta",    role: "Deep Learning (BiLSTM)"   },
  { name: "Sneha Iyer",     role: "BERT Fine-Tuning"         },
  { name: "Karan Verma",    role: "Backend & Flask API"      },
  { name: "Karan Verma",   role: "Frontend Developer"       },
  { name: "Dev Joshi",      role: "Data Collection & EDA"    },
  { name: "Meera Nair",     role: "Model Evaluation"         },
  { name: "Arjun Gupta",    role: "SVM & Baseline Models"    },
  { name: "Tanya Bose",     role: "Documentation & Testing"  },
  { name: "Smignita Roy",   role: "Student Contributor"       },
];

const MENTOR = {
  name:  "Dr. Rajesh Kumar",
  title: "Associate Professor\nDept. of Computer Science & Engineering",
  bio:   "Dr. Kumar specializes in Natural Language Processing and Machine Learning. He has guided numerous research projects in misinformation detection and computational linguistics. His mentorship was invaluable throughout this project.",
};

/* ── HELPER ── */
function getInitials(name) {
  return name.split(" ").map((n) => n[0]).join("").slice(0, 2).toUpperCase();
}

/* ── PAGE ── */
export default function About() {
  return (
    <main className="about-page">

      {/* ── HERO ── */}
      <div className="about-hero">
        <div className="about-hero-badge">
          <span>📰</span> Term Paper Project
        </div>
        <h1>About the Project</h1>
        <p>
          A fake news detection system built using classical Machine Learning,
          Deep Learning, and Transformer-based models — trained on the ISOT
          Fake News Dataset.
        </p>
      </div>

      {/* ── HOW IT WORKS ── */}
      <div className="about-section">
        <div className="section-label">
          <span>🧠</span> Mechanism
        </div>
        <h2 className="section-title">How It Works</h2>
        <p className="section-subtitle">
          The system processes raw news articles through a multi-stage pipeline
          before classifying them as real or fake.
        </p>

        <div className="mechanism-grid">
          {MECHANISM.map((item) => (
            <div className="mechanism-card" key={item.title}>
              <div className="mechanism-icon">{item.icon}</div>
              <h3>{item.title}</h3>
              <p>{item.description}</p>

              {item.extra === "models" && (
                <div className="model-chips-list">
                  {MODELS_LIST.map((m) => (
                    <span
                      key={m.label}
                      className={`model-chip-static ${m.highlight ? "highlight" : ""}`}
                    >
                      {m.label}
                    </span>
                  ))}
                </div>
              )}

              {item.extra === "pipeline" && (
                <div className="pipeline">
                  {PIPELINE_STEPS.map((step, i) => (
                    <span key={step} style={{ display: "flex", alignItems: "center", gap: 6 }}>
                      <span className="pipeline-step">{step}</span>
                      {i < PIPELINE_STEPS.length - 1 && (
                        <span className="pipeline-arrow">→</span>
                      )}
                    </span>
                  ))}
                </div>
              )}
            </div>
          ))}
        </div>
      </div>

      {/* ── TEAM ── */}
      <div className="about-section">
        <div className="section-label">
          <span>👥</span> The Team
        </div>
        <h2 className="section-title">Meet Our Team</h2>
        <p className="section-subtitle">
          A group of {TEAM.length} passionate students who built this project
          from scratch.
        </p>

        <div className="team-grid">
          {TEAM.map((member, index) => (
            <div className="team-card" key={member.name}>
              <div className="team-avatar-placeholder">
                {getInitials(member.name)}
              </div>
              <div className="team-number">Member {String(index + 1).padStart(2, "0")}</div>
              <div className="team-name">{member.name}</div>
              <div className="team-role">{member.role}</div>
              <div className="team-socials">
                <a href="#" className="social-link" target="_blank" rel="noreferrer">GH</a>
                <a href="#" className="social-link" target="_blank" rel="noreferrer">in</a>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* ── MENTOR ── */}
      <div className="about-section">
        <div className="section-label">
          <span>🎓</span> Guidance
        </div>
        <h2 className="section-title">Our Mentor</h2>
        <p className="section-subtitle">
          This project was completed under the expert guidance of our mentor.
        </p>

        <div className="mentor-wrapper">
          <div className="mentor-card">
            <div className="mentor-left">
              <div className="mentor-avatar-placeholder">
                {getInitials(MENTOR.name)}
              </div>
              <span className="mentor-badge">⭐ Project Mentor</span>
            </div>

            <div className="mentor-right">
              <div className="mentor-name">{MENTOR.name}</div>
              <div className="mentor-title">{MENTOR.title}</div>
              <hr className="mentor-divider" />
              <p className="mentor-bio">{MENTOR.bio}</p>
              <div className="team-socials">
                <a href="#" className="social-link" target="_blank" rel="noreferrer">in</a>
                <a href="#" className="social-link" target="_blank" rel="noreferrer">GH</a>
              </div>
            </div>
          </div>
        </div>
      </div>

    </main>
  );
}