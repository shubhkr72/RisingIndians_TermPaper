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
  { name: "Shubham Kumar",   role: "Team Lead & ML Engineer & Full Stack Web Developer"  },
  { name: "Manish Kumar",    role: "Full Stack Web Developer" },
  { name: "Prashant Kumar Singh",    role: "Model Evaluation & Documentation"   },
  { name: "Ritesh Kumar",     role: "Developer"         },
  { name: "Mohit Tiwari",    role: "Developer"      },
  { name: "Akshat Dubey",   role: "Developer"       },
  { name: "Chanchlesh",      role: "Developer"    },
  { name: "Anjali Kushwaha",     role: "Developer" },
  { name: "Smignita Roy",    role: "Developer"    },
  { name: "Aadarsh Munna",     role: "Data Collection & Testing"  },

];

const MENTOR = {
  name:  "Dr. Suman Deb",
  title: "Assistant Professor\nDept. of Computer Science & Engineering\nNational Institute of Technology,Agartala",
  bio:   "whose research focuses on Human-Computer Interaction and Educational Technologies, with a strong emphasis on innovative, interactive learning methods. He leads the HCI and AI Lab, develops low-cost technological solutions, and actively works on transforming traditional classrooms while mentoring motivated students in cutting-edge research.",
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
          A group of passionate students who built this project
          from scratch.
        </p>

        <div className="team-grid">
          {TEAM.map((member, index) => (
            <div className="team-card" key={member.name}>
              <div className="team-avatar-placeholder">
                {getInitials(member.name)}
              </div>
              <div className="team-name">{member.name}</div>
              <div className="team-role">{member.role}</div>
              {/* <div className="team-socials">
                <a href="#" className="social-link" target="_blank" rel="noreferrer">GH</a>
                <a href="#" className="social-link" target="_blank" rel="noreferrer">in</a>
              </div> */}
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
              <img 
                src="/images/mentor.jpeg"   // path to your image
                alt={MENTOR.name}
                className="mentor-avatar"
              />
            </div>

            <div className="mentor-right">
              <div className="mentor-name">{MENTOR.name}</div>
              <div className="mentor-title">{MENTOR.title}</div>
              <hr className="mentor-divider" />
              <p className="mentor-bio">{MENTOR.bio}</p>
              {/* <div className="team-socials">
                <a href="#" className="social-link" target="_blank" rel="noreferrer">in</a>
                <a href="#" className="social-link" target="_blank" rel="noreferrer">GH</a>
              </div> */}
            </div>
          </div>
        </div>
      </div>

    </main>
  );
}