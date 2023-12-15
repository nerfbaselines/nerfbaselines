import { Button } from '@mantine/core';
import Image from 'next/image'
import { useMemo } from 'react';
import MainHeader from './components/MainHeader';
import { IconExternalLink } from '@tabler/icons-react';
import DatasetResultsTable from '@/app/components/DatasetResultsTable';
import Link from 'next/link';
import * as api from '@/utils/api';
import logo from "@/public/logo.png";

export default async function Home() {
  const datasets = await api.getDatasets();
  const datasetsResults = await Promise.all(datasets.map(async ({ id }) => api.getDatasetData(id)));
  return (
    <main>
      <MainHeader />
      {datasetsResults.map((datasetResults) => (
        <div key={datasetResults.id} className="result-set-panel">
          <div className="result-set-panel__header">
            <h2>
              <Link href={`/${datasetResults.id}`}>{datasetResults.name}&nbsp;<span className="text-blue"><IconExternalLink size={20} /></span></Link>
            </h2>
            <p className="text-justify">{datasetResults.description}</p>
          </div>
          <DatasetResultsTable 
            datasetId={datasetResults.id} 
            methodResults={datasetResults.methods} 
            metrics={datasetResults.metrics}
            scenes={datasetResults.scenes} />
        </div>
      ))}
    </main>
  );
}
// <Button variant="filled" radius="xl">github</Button>