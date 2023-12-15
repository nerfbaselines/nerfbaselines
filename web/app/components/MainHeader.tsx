"use client";
import { Button } from '@mantine/core';
import Image from 'next/image'
import { IconExternalLink, IconBrandGithubFilled } from '@tabler/icons-react';
import Link from 'next/link';
import logo from "@/public/logo.png";
import styles from './MainHeader.module.css';

export default function MainHeader() {
  return (
      <header className="main-header">
        <div className="main-header__top-row">
          <img src={logo.src} alt="NerfBaselines" />
          <h1>
            <span>Nerf<br/>Baselines</span>
          </h1>
        </div>
        <div className="main-header__buttons-row">
          <Button variant="filled" radius="xl" component={Link} href="https://github.com/jkulhanek/nerfbaselines"><IconBrandGithubFilled size={18} /> GitHub</Button>
        </div>
        <p className="main-header__abstract">
          NerfBaselines is a framework for <strong>evaluating and comparing existing NeRF methods</strong>.
          Currently, most official implementations use different dataset loaders, evaluation protocols, and metrics which renders the comparison of methods difficult.
          Therefore, this project aims to provide a <strong>unified interface</strong> for running and evaluating methods on different datasets in a consistent way using the same metrics. But instead of reimplementing the methods, <strong>we use the official implementations</strong> and wrap them so that they can be run easily using the same interface.
        </p>
      </header>
  );
}