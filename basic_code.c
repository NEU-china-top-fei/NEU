//神经网络基础版，预测粮食产量
#include<stdio.h>
#include<math.h>
#include<stdlib.h>
#define learn_rate 0.02
double operation(double *we,double *wa,double *sita,double *e);
//定义sigmoid函数
double sigmoid(double x)
{
    double y;
    y=1.0/(1.0+exp(-x));
    return y;
}

int main()
{
    //评价
    double *e=(double*)malloc(sizeof(double));
    //初始参数
    double we=0.5,wa=0.5,sita=1.0;
    for(int a=0;a<10;a++)
    {
        //调用函数进行训练
        operation(&we,&wa,&sita,e);

        printf("%lf\n",we);
        printf("%lf\n",wa);
        printf("%lf\n",sita);
        //调用函数评估
        operation(&we,&wa,&sita,e);
        printf("%lf\n",*e);
        
    }
    free(e);
    
    
   

    return 0;
}
//请用c语言编写一段代码实现只有一个神经元的最简单的神经网络及其训练过程。（通过有效灌溉和农用化肥预测粮食产量）
double operation(double *we,double *wa,double *sita,double *e)
{
    *e=0;
    //影响因素:有效灌溉，农用化肥。
    double factor[100][2];
    //实际值
    double production[100];
    //预测值
    double production_p,grad,grad_sum;
    //读取数据
    FILE*fp=fopen("data_input.txt","r");
    if(fp==NULL)
    {
        printf("无文件");
        return -1;
    }
    for(int j=0;j<100;j++)
    {
        for(int i=0;i<2;i++)
        {
            fscanf(fp,"%lf",&factor[j][i]);
            
        }
        fscanf(fp,"\n");
    }
    fclose(fp);

    FILE*fpq=fopen("data_output.txt","r");
    if(fpq==NULL)
    {
        printf("无文件");
        return -1;
    }
    for(int j=0;j<100;j++)
    {
        
        fscanf(fpq,"%lf",&production[j]);   
    }
    fclose(fpq);
    //归一化
    for (int j = 0; j < 100; j++) 
    {
        factor[j][0] = factor[j][0] / 100000.0; 
        factor[j][1] = factor[j][1] / 10000.0;  
        production[j] = production[j] / 10000.0; 
    }

    //训练
    for(int j=0;j<100;j++)
    {
        
        production_p = sigmoid(factor[j][0] * *we + factor[j][1] * *wa - *sita);
        grad = production_p * (1.0 - production_p) * (production[j] - production_p); 
        *e=(production[j]-production_p)*(production[j]-production_p);
        grad_sum+=grad;
        //调试用代码，已注释：printf("production_p: %lf, production[j]: %lf, grad: %lf\n", production_p, production[j], grad);
    }
    //更新参数
    for(int j=0;j<100;j++)
    {
        double we1=factor[j][0]*grad_sum;
        double wa1=factor[j][1]*grad_sum;
        double sita1=-grad_sum;
        *we=*we+learn_rate*we1;
        *wa=*wa+learn_rate*wa1;
        *sita=*sita+learn_rate*sita1;
    }
        

}