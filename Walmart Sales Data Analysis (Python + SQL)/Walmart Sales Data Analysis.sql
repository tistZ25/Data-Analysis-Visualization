use walmart_db;
select * from walmart;

select count(*) from walmart;

-- Distinct Payment Type
select distinct payment_method from walmart;

-- Distinct no. of branches
select 
	branch, 
	sum(quantity)
from walmart group by branch;

-- No. of transactions from each payment method
select 
    payment_method,
    count(*)
from walmart group by payment_method;

-- Distinct Categories of goods
select distinct category from walmart;

-- Number of Transactions made in each product category
select 
    category,
    count(*)
from walmart group by category;

-- No. of different stores
select count(distinct Branch) from walmart;

-- Maximum quantity purchased
select max(quantity) from walmart;

-- Solving the Business Problems

/* 1. Analyze Payment Methods and Sales
 ● Question: What are the different payment methods, and how many transactions and
 items were sold with each method?
 ● Purpose: This helps Walmart track sales volume by payment type, providing insights
 into customer purchasing habits, aiding in payment optimization strategies.*/
 
select 
    payment_method,
    count(*) as no_of_transactions,
    sum(quantity) as sales_volume
from walmart group by payment_method;

/* 2. Identify the Highest-Rated Category in Each Branch
 ● Question: Which category received the highest average rating in each branch?
 ● Purpose: This allows Walmart to recognize and promote popular categories in specific
 branches, enhancing customer satisfaction and branch-specific marketing. 
*/

select * from
(select 
	Branch, 
    category, 
    AVG(rating),
    RANK() OVER (PARTITION BY Branch ORDER BY AVG(rating) DESC) as rank1
from walmart group by Branch, category) AS ranked_data 
where rank1 = 1;

/* 3. Determine the Busiest Day for Each Branch
 ● Question: What is the busiest day of the week for each branch based on transaction
 volume?
 ● Purpose: This insight helps in optimizing staffing and inventory management to
 accommodate peak days. */
 
select * from
	(select 
		branch,
        DATE_FORMAT(STR_TO_DATE(date, '%d/%m/%y'), '%W') as week_day,
		count(*) as transation_volume,
		rank() over(partition by Branch order by count(*) desc) as rank2
	from walmart 
    group by branch, week_day) as ranked_data1
where rank2 = 1;

 /* 4. Analyze Category Ratings by City
 ● Question: What are the average, minimum, and maximum ratings for each category in
 each city?
 ● Purpose: This data can guide city-level promotions, allowing Walmart to address
 regional preferences and improve customer experiences. */
 
select 
    City,
    category,
    avg(rating),
    min(rating),
    max(rating)
from walmart group by City, category order by City;
 
 /* 5. Calculate Total Profit by Category
 ● Question: What is the total profit for each category, ranked from highest to lowest?
 ● Purpose: Identifying high-profit categories helps focus efforts on expanding these
 products or managing pricing strategies effectively. */
 
select
	category,
    sum(profit_margin * quantity) as total_profit,
    sum(total) as total_revenue
from walmart
group by category order by total_profit desc;
 
/* 6. Determine the Most Common Payment Method per Branch
 ● Question: What is the most frequently used payment method in each branch?
 ● Purpose: This information aids in understanding branch-specific payment preferences,
 potentially allowing branches to streamline their payment processing systems. */

select * from
	(select
		Branch,
		payment_method,
		count(*) no_of_transactions,
		rank() over(partition by Branch order by count(*) desc) as rank3
	from walmart
	group by Branch, payment_method) as ranked_data2
where rank3 = 1;
 
/* 7. Analyze Sales Shifts Throughout the Day
 ● Question: How many transactions occur in each shift (Morning, Afternoon, Evening)
 across branches?
 ● Purpose: This insight helps in managing staff shifts and stock replenishment schedules,
 especially during high-sales periods. */
 
select
	Branch,
	case 
		when hour(time(time)) < 12 then 'Morning'
		when hour(time(time)) between 12 and 17 then 'Afternoon'
		else "Evening"
	End shift_time,
	count(*)
from walmart
group by Branch, shift_time
order by Branch, count(*) desc;

/* 8. Identify Branches with Highest Revenue Decline Year-Over-Year
 ● Question: Which branches experienced the largest decrease in revenue compared to
 the previous year?
 ● Purpose: Detecting branches with declining revenue is crucial for understanding
 possible local issues and creating strategies to boost sales or mitigate losses. */
 
 -- Decrease Revenue in Ratio (Current Year 2023, Last Year 2022)

with revenue2022 as
(select 
	Branch,
    sum(Total) as Revenue_2022
from walmart 
where YEAR(STR_TO_DATE(date, '%d/%m/%y')) = 2022
group by Branch),

revenue2023 as
(select 
	Branch,
    sum(Total) as Revenue_2023
from walmart 
where YEAR(STR_TO_DATE(date, '%d/%m/%y')) = 2023
group by Branch)

select 
	Y2022.Branch,
    Y2022.Revenue_2022 as Revenue_2022,
    Y2023.Revenue_2023 as Revenue_2023,
    round((Y2022.Revenue_2022-Y2023.Revenue_2023)*100/Y2022.Revenue_2022, 2) as rev_dec_ratio
from revenue2022 as Y2022
join
revenue2023 as Y2023
on Y2022.Branch = Y2023.Branch

where Revenue_2022 > Revenue_2023
order by rev_dec_ratio desc
limit 5;


 
 
 
 
 
 
 
 
 