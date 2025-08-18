import React from 'react';
import ComponentCreator from '@docusaurus/ComponentCreator';

export default [
  {
    path: '/__docusaurus/debug',
    component: ComponentCreator('/__docusaurus/debug', '826'),
    exact: true
  },
  {
    path: '/__docusaurus/debug/config',
    component: ComponentCreator('/__docusaurus/debug/config', '404'),
    exact: true
  },
  {
    path: '/__docusaurus/debug/content',
    component: ComponentCreator('/__docusaurus/debug/content', '818'),
    exact: true
  },
  {
    path: '/__docusaurus/debug/globalData',
    component: ComponentCreator('/__docusaurus/debug/globalData', '5d0'),
    exact: true
  },
  {
    path: '/__docusaurus/debug/metadata',
    component: ComponentCreator('/__docusaurus/debug/metadata', '49a'),
    exact: true
  },
  {
    path: '/__docusaurus/debug/registry',
    component: ComponentCreator('/__docusaurus/debug/registry', 'a1a'),
    exact: true
  },
  {
    path: '/__docusaurus/debug/routes',
    component: ComponentCreator('/__docusaurus/debug/routes', '1e4'),
    exact: true
  },
  {
    path: '/',
    component: ComponentCreator('/', '0f1'),
    exact: true
  },
  {
    path: '/',
    component: ComponentCreator('/', '5e1'),
    routes: [
      {
        path: '/',
        component: ComponentCreator('/', 'f50'),
        routes: [
          {
            path: '/',
            component: ComponentCreator('/', 'f7d'),
            routes: [
              {
                path: '/features',
                component: ComponentCreator('/features', 'e44'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/introduction',
                component: ComponentCreator('/introduction', 'e7c'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/user-guide',
                component: ComponentCreator('/user-guide', '96b'),
                exact: true,
                sidebar: "tutorialSidebar"
              }
            ]
          }
        ]
      }
    ]
  },
  {
    path: '*',
    component: ComponentCreator('*'),
  },
];
