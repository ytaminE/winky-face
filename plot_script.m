filename = '2.xlsx';
sheet = 1;
xlRange = 'B2:W10';
A = xlsread(filename,sheet,xlRange);
A = fliplr(A);
x=1:11;

cpu_bench_build=A(1,2:2:22);
gpu_bench_build=A(2,2:2:22);
array_offset_build=A(3,2:2:22);
SOA_build=A(5,2:2:22);
matrix_sparse_build=A(6,2:2:22);
matrix_build=A(7,2:2:22);
matrix_cpu_build=A(8,2:2:22);

cpu_bench_calc=A(1,1:2:21);
gpu_bench_calc=A(2,1:2:21);
array_offset_calc=A(3,1:2:21);
SOA_calc=A(5,1:2:21);
matrix_sparse_calc=A(6,1:2:21);
matrix_calc=A(7,1:2:21);
matrix_cpu_calc=A(8,1:2:21);

figure(1)
plot(x,cpu_bench_build,'linewidth',1.5);
hold on;
plot(x,gpu_bench_build,'linewidth',1.5);
hold on;
plot(x,array_offset_build,'linewidth',1.5);
hold on;
plot(x,SOA_build,'linewidth',1.5);
hold on;
plot(x,matrix_sparse_build,'linewidth',1.5);
hold on;
plot(x,matrix_build,'linewidth',1.5);
hold on;
plot(x,matrix_cpu_build,'linewidth',1.5);
title('Build Time')
xlabel('Data Set Size')
ylabel('Time(s)')
legend('cpu bench','gpu bench','array offset','SOA','matrix sparse','matrix gpu','matrix cpu')
x_Matrix={'1000x50','1000x100','1000x200','10000x50','10000x100','10000x200','100000x50','100000x100','100000x200','100000x50','100000x100'};
set(gca,'xticklabel',x_Matrix);

figure(2)
plot(x,cpu_bench_calc,'linewidth',1.5);
hold on;
plot(x,gpu_bench_calc,'linewidth',1.5);
hold on;
plot(x,array_offset_calc,'linewidth',1.5);
hold on;
plot(x,SOA_calc,'linewidth',1.5);
hold on;
plot(x,matrix_sparse_calc,'linewidth',1.5);
hold on;
plot(x,matrix_calc,'linewidth',1.5);
hold on;
plot(x,matrix_cpu_calc,'linewidth',1.5);
title('Calculation Time')
xlabel('Data Set Size')
ylabel('Time(s)')
legend('cpu bench','gpu bench','array offset','SOA','matrix sparse','matrix gpu','matrix cpu')
x_Matrix={'1000x50','1000x100','1000x200','10000x50','10000x100','10000x200','100000x50','100000x100','100000x200','100000x50','100000x100'};
set(gca,'xticklabel',x_Matrix);


