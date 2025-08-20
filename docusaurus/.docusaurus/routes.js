import React from 'react';
import ComponentCreator from '@docusaurus/ComponentCreator';

export default [
  {
    path: '/user-guide/',
    component: ComponentCreator('/user-guide/', 'ea9'),
    exact: true
  },
  {
    path: '/user-guide/',
    component: ComponentCreator('/user-guide/', '384'),
    routes: [
      {
        path: '/user-guide/',
        component: ComponentCreator('/user-guide/', '200'),
        routes: [
          {
            path: '/user-guide/',
            component: ComponentCreator('/user-guide/', 'a7e'),
            routes: [
              {
                path: '/user-guide/features',
                component: ComponentCreator('/user-guide/features', '6a1'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/user-guide/introduction',
                component: ComponentCreator('/user-guide/introduction', 'a80'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/user-guide/user-guide',
                component: ComponentCreator('/user-guide/user-guide', '692'),
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
