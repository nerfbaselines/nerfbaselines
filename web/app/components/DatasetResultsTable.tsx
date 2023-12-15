"use client";

import * as api from '@/utils/api';
import { useMemo } from 'react';
import {
  MantineReactTable,
  useMantineReactTable,
  type MRT_ColumnDef,
  type MRT_Cell,
} from 'mantine-react-table';
import Link from 'next/link';

export default function DatasetResultsTable({
  datasetId,
  methodResults,
  metrics,
  metricDetail,
  scenes,
  showMethod,
}: {
  datasetId: string,
  methodResults: api.MethodResults[],
  metrics: api.Metric[],
  scenes: api.Scene[],
  metricDetail?: string,
  showMethod?: boolean,
}) {
  const columns = useMemo<MRT_ColumnDef<api.MethodResults>[]>(
    () => {
      let list: MRT_ColumnDef<api.MethodResults>[] = [];
      if (showMethod ?? true) list.push({
          accessorKey: 'name',
          header: 'Method',
          Cell: ({ cell }: { cell: MRT_Cell<api.MethodResults> }) => (
            <Link href={`/${cell.getContext().row.original.id}`}>{cell.getValue() as any}</Link>
          ),
        });
      if (metricDetail !== undefined) {
        return list.concat(scenes.map((scene) => ({
          accessorKey: `scenes.${scene.id}.${metricDetail}`,
          header: scene.name ?? scene.id,
          size: 30,
          minSize: 30,
          enableClickToCopy: true,
        })));
      } else {
        return list.concat(metrics.map((metric) => ({
          accessorKey: `${metric.id}`,
          header: metric.name,
          size: 30,
          minSize: 30,
          enableClickToCopy: true,
        })));
      }
    },
    [scenes, metrics]
  );
  const table = useMantineReactTable({
    columns,
    data: methodResults,
    enableColumnActions: false,
    enableColumnFilters: false,
    enablePagination: false,
    enableDensityToggle: false,
    enableSorting: true,
    enableBottomToolbar: false,
    enableTopToolbar: false,
    initialState: {
      density: "xs",
    },
    mantineTableProps: {
      withColumnBorders: false,
      withBorder: false,
    }
  });
  return (
    <MantineReactTable table={table} />
  );
}