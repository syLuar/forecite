import React from 'react';
import ComponentCreator from '@docusaurus/ComponentCreator';

export default [
  {
    path: '/docs/',
    component: ComponentCreator('/docs/', 'a6b'),
    exact: true
  },
  {
    path: '/docs/',
    component: ComponentCreator('/docs/', '209'),
    routes: [
      {
        path: '/docs/',
        component: ComponentCreator('/docs/', '7e5'),
        routes: [
          {
            path: '/docs/',
            component: ComponentCreator('/docs/', 'e3c'),
            routes: [
              {
                path: '/docs/features',
                component: ComponentCreator('/docs/features', '073'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/introduction',
                component: ComponentCreator('/docs/introduction', '457'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/user-guide',
                component: ComponentCreator('/docs/user-guide', 'e20'),
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
