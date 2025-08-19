// docusaurus.config.ts
import type { Config } from '@docusaurus/types';
import { themes as prismThemes } from 'prism-react-renderer';

const config: Config = {
  title: 'Forecite Documentation',
  tagline: 'AI-Powered Legal Research Assistant',
  favicon: 'img/favicon.ico',
  url: 'https://forecite.site',
  baseUrl: '/user-guide/', 
  organizationName: 'forecite',
  projectName: 'forecite-docs',

  onBrokenLinks: 'throw',
  onBrokenMarkdownLinks: 'warn',

  i18n: {
    defaultLocale: 'en',
    locales: ['en'],
  },

  presets: [
    [
      'classic',
      {
        docs: {
          sidebarPath: './sidebars.js',
          routeBasePath: '/',
        },
        theme: {
          customCss: './src/css/custom.css',
        },
        blog: false,
      },
    ],
  ],

  themeConfig: {
    colorMode: {
      defaultMode: 'light',
      disableSwitch: true,
    },
    navbar: {
      title: 'Forecite',
      logo: {
        alt: 'Forecite Logo',
        src: 'img/logo.png',
      },
      items: [
        {
          type: 'doc',
          docId: 'introduction',
          position: 'left',
          label: 'Documentation',
        },
        {
          href: 'https://forecite.site',
          label: 'Launch App',
          position: 'right',
        },
      ],
    },
    footer: {
      style: 'dark',
      logo: {
        alt: 'Forecite Logo',
        src: 'img/logo.png',
        href: '/',
        width: 120,
      },
      links: [
        {
          title: 'Documentation',
          items: [
            {
              label: 'Getting Started',
              to: '/introduction',
            },
            {
              label: 'User Guide',
              to: '/user-guide',
            },
            {
              label: 'Features',
              to: '/features',
            },
          ],
        },
        {
          title: 'Application',
          items: [
            {
              label: 'Launch Forecite',
              href: 'https://forecite.site',
            },
          ],
        },
      ],
      copyright: `Copyright © ${new Date().getFullYear()} Forecite.`,
    },
    prism: {
      theme: prismThemes.github,
      darkTheme: prismThemes.dracula,
    },
  } satisfies import('@docusaurus/preset-classic').ThemeConfig,
};

export default config;


