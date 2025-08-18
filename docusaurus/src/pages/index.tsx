// src/pages/index.tsx
import React from 'react';
import clsx from 'clsx';
import Link from '@docusaurus/Link';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import Layout from '@theme/Layout';
import styles from './index.module.css';

interface FeatureProps {
  title: string;
  description: string;
  emoji: string;
}

function Feature({ title, description, emoji }: FeatureProps) {
  return (
    <div className={styles.feature}>
      <div className={styles.featureIcon}>{emoji}</div>
      <h3>{title}</h3>
      <p>{description}</p>
    </div>
  );
}

function HomepageHeader() {
  const {siteConfig} = useDocusaurusContext();
  return (
    <header className={clsx('hero', styles.heroBanner)}>
      <div className="container">
        <img 
          src="img/logo.png" 
          alt="Forecite Logo" 
          className={styles.heroLogo}
        />
        <h1 className="hero__title">Forecite User Guide</h1>
        <p className="hero__subtitle">Comprehensive guide to using Forecite's AI-powered legal research tools</p>
        
        <div className={styles.buttons}>
          <Link
            className={clsx('button button--primary button--lg')}
            to="/introduction">
            Get Started
          </Link>
        </div>
      </div>
    </header>
  );
}

function FeaturesSection() {
  const features = [
    {
      title: 'Legal Search',
      description: 'Find relevant cases, statutes, and legal documents with AI-powered search',
      emoji: '🔍',
    },
    {
      title: 'Case Management',
      description: 'Organize and track your legal cases with comprehensive tools',
      emoji: '📁',
    },
    {
      title: 'AI Analysis',
      description: 'Get intelligent insights and analysis on legal documents',
      emoji: '🤖',
    },
    {
      title: 'Argument Drafting',
      description: 'Build and refine legal arguments with AI assistance',
      emoji: '⚖️',
    },
  ];

  return (
    <section className={styles.features}>
      <div className="container">
        <h2>Key Features</h2>
        <div className={styles.featuresGrid}>
          {features.map((feature, idx) => (
            <Feature key={idx} {...feature} />
          ))}
        </div>
      </div>
    </section>
  );
}

export default function Home(): JSX.Element {
  const {siteConfig} = useDocusaurusContext();
  return (
    <Layout
      title={siteConfig.title}
      description="Learn how to use Forecite's AI-powered legal research tools">
      <HomepageHeader />
      <main>
        <FeaturesSection />
      </main>
    </Layout>
  );
}
