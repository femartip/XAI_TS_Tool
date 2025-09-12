import React from 'react';

const AboutPage = ({ onBack }) => {
  return (
    <div className="about-page">
      <button className="button-nav" onClick={onBack} style={{ margin: '10px' }}>
        Back to Home
      </button>

      <h1>About the Interactive XAI Tool for Time Series</h1>

      <section className="about-section">
        <h2>Introduction</h2>
        <p>
          This tool improves the interpretability of time series classifiers (TSCs) by generating
          loyal simplifications as explanations. Simplification is central in explainable AI (XAI) for TSC: it
          reduces noise and makes patterns easier to understand. However, choosing how much to
          simplify is tricky—too much can remove class-relevant features, too little reduces
          interpretability. We address this with loyalty: the probability that a classifier’s prediction
          remains unchanged after simplification. Loyalty is an intuitive, measurable criterion for
          controlling simplification. The tool automatically finds the most simplified representation of a
          time series that satisfies a user-specified loyalty threshold. It supports custom datasets and
          models, demonstrating how loyalty-based simplifications enhance interpretability in practice.
        </p>
      </section>

      <section className="about-section">
        <h2>Core Ideas</h2>
        <ul>
          <li>
            <b>Complexity:</b> measures how simple a series is, defined by the number of straight-line
            segments (or selected points) used in the simplification.
          </li>
          <li>
            <b>Loyalty:</b> the fraction of cases where the classifier’s prediction does not change after
            simplification. Higher loyalty means the simplification better preserves the model’s decision.
          </li>
          <li>
            <b>Alpha (α):</b> controls the degree of simplification under the hood. α = 1 means no
            simplification; α = 0 gives a single segment (maximum simplification). The interface lets you
            steer α indirectly using loyalty or complexity targets.
          </li>
        </ul>
      </section>

      <section className="about-section">
        <h2>Using the Interface</h2>
        <ol>
          <li>
            <b>Select a dataset and instance:</b> use built-in datasets or upload your own on the Import
            page. Pick a specific time series instance to explain.
          </li>
          <li>
            <b>Pick a simplification method:</b> choose among the available algorithms.
          </li>
          <li>
            <b>Choose control mode:</b> set either a target <i>Loyalty</i> (κ) or <i>Complexity</i>. The tool will
            compute and display the corresponding simplification.
          </li>
          <li>
            <b>Inspect the chart:</b> compare the Original and Simplification curves. The legend and colors
            reflect class information to aid interpretation.
          </li>
          <li>
            <b>Adjust and iterate:</b> tune thresholds or switch methods to balance fidelity vs. simplicity
            for your audience—experts may prefer high loyalty; end-users may prefer low complexity.
          </li>
        </ol>
      </section>

      <section className="about-section">
        <h2>What You Can Do</h2>
        <ul>
          <li><b>Explore prototypes and simplifications</b> to reveal class-relevant patterns.</li>
          <li><b>Tailor explanations</b> for different stakeholders by adjusting loyalty or complexity.</li>
          <li><b>Use your own data and models</b> to generate faithful, simplified views of predictions.</li>
        </ul>
      </section>

      <section className="about-section">
        <h2>Takeaway</h2>
        <p>
          The tool helps you navigate the trade-off between simple, readable time series and faithful
          representations of your model’s decisions. By controlling loyalty and complexity, you can craft
          explanations that are both trustworthy and easy to understand.
        </p>
      </section>
    </div>
  );
};

export default AboutPage;

